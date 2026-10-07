"""CJ1 denoiser: a Keras autoencoder that predicts the speech mask M from the noisy magnitude |Y|.

Same pipeline as signal-path.html: 16 kHz, Hann window, hop = window/4,
mask target M = |S|^2 / (|S|^2 + |N|^2), output = ISTFT(M * |Y| * e^(j angle Y)).

  python denoise.py train --clean DIR --noise DIR [--out cj1.keras]
  python denoise.py enhance noisy.wav clean.wav [--model cj1.keras]
  python denoise.py demo        # self-check on synthetic speech, no data needed
"""
import argparse, glob, os
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, resample_poly

SR, WIN = 16000, 512
HOP = WIN // 4
CTX = 5  # frames of context on each side of the frame being masked
SEG = 2 * SR  # training clip length, matches the page's 2 s clip


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return resample_poly(x, SR, sr).astype("float32") if sr != SR else x


def spec(x):
    return stft(x, SR, window="hann", nperseg=WIN, noverlap=WIN - HOP)[2].T  # (frames, bins)


def features(Y):
    """log|Y| with CTX neighbouring frames stacked onto each frame -> (frames, (2*CTX+1)*bins)."""
    L = np.log(np.abs(Y) + 1e-6).astype("float32")
    P = np.pad(L, ((CTX, CTX), (0, 0)), mode="edge")
    return np.concatenate([P[i:i + len(L)] for i in range(2 * CTX + 1)], axis=1)


def mix(s, n, snr_db):
    n = n * np.sqrt(np.sum(s**2) / (np.sum(n**2) + 1e-12) / 10 ** (snr_db / 10))
    return s + n, n


def crop(x, rng):
    if len(x) < SEG:
        x = np.tile(x, SEG // len(x) + 1)
    i = rng.integers(0, len(x) - SEG + 1)
    return x[i:i + SEG]


def make_pairs(clean, noise, count, rng, snr=(-5, 20)):
    """Random clean clip + random noise clip at a random SNR -> (features, oracle mask)."""
    X, M = [], []
    for _ in range(count):
        s = crop(clean[rng.integers(len(clean))], rng)
        y, n = mix(s, crop(noise[rng.integers(len(noise))], rng), rng.uniform(*snr))
        S2, N2 = np.abs(spec(s)) ** 2, np.abs(spec(n)) ** 2
        X.append(features(spec(y)))
        M.append((S2 / (S2 + N2 + 1e-12)).astype("float32"))
    return np.concatenate(X), np.concatenate(M)


def build(X):
    import keras
    from keras import layers
    norm = layers.Normalization()
    norm.adapt(X)  # stores mean/std inside the model, so enhance needs nothing else
    bins = WIN // 2 + 1
    model = keras.Sequential([
        keras.Input((X.shape[1],)), norm,
        layers.Dense(1024, activation="relu"), layers.Dropout(0.2),
        layers.Dense(256, activation="relu"),  # bottleneck
        layers.Dense(1024, activation="relu"), layers.Dropout(0.2),
        layers.Dense(bins, activation="sigmoid"),  # gain in [0, 1] per bin
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train(clean, noise, out, clips=2000, epochs=30, rng=None):
    import keras
    rng = rng or np.random.default_rng(0)
    X, M = make_pairs(clean, noise, clips, rng)
    print(f"{clips} clips -> {len(X):,} frames")
    model = build(X)
    model.fit(X, M, batch_size=256, epochs=epochs, validation_split=0.1, shuffle=True,
              callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    model.save(out)
    print("saved", out)
    return model


def enhance(model, y):
    Y = spec(y)
    Mhat = model.predict(features(Y), batch_size=1024, verbose=0)
    s = istft((Mhat * Y).T, SR, window="hann", nperseg=WIN, noverlap=WIN - HOP)[1]
    return s[:len(y)].astype("float32")


def snr(ref, est):
    return 10 * np.log10(np.sum(ref**2) / (np.sum((ref - est) ** 2) + 1e-12))


def synth_voice(rng, dur=4.0):
    """Harmonic 'voice' like the page: gliding pitch, formant envelope, syllable bursts."""
    t = np.arange(int(dur * SR)) / SR
    f0 = rng.uniform(100, 220) * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.5, 2) * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    formants = rng.uniform([500, 1200, 2400], [900, 2000, 3200])
    x = sum(np.sin(k * phase) * sum(np.exp(-((k * f0 - F) / 150) ** 2) for F in formants) / k**0.5
            for k in range(1, 30))
    env = np.clip(np.sin(2 * np.pi * rng.uniform(2, 5) * t + rng.uniform(0, 6)), 0, None) ** 0.5
    return (x * env).astype("float32")


def synth_noise(rng, dur=4.0):
    n = rng.standard_normal(int(dur * SR))
    if rng.random() < 0.5:  # pink-ish
        n = np.cumsum(n); n -= np.convolve(n, np.ones(64) / 64, "same")
    if rng.random() < 0.3:  # mains hum
        t = np.arange(len(n)) / SR
        n = n / n.std() + 3 * sum(np.sin(2 * np.pi * 50 * k * t) / k for k in range(1, 6))
    return n.astype("float32")


def demo():
    """Self-check: train briefly on synthetic data, the model must clean a held-out clip."""
    rng = np.random.default_rng(1)
    clean = [synth_voice(rng) for _ in range(40)]
    noise = [synth_noise(rng) for _ in range(20)]
    model = train(clean, noise, "demo.keras", clips=300, epochs=8, rng=rng)
    s = crop(synth_voice(rng), rng)
    y, _ = mix(s, crop(synth_noise(rng), rng), 0)
    before, after = snr(s, y), snr(s, enhance(model, y))
    print(f"held-out clip: SNR {before:.1f} dB -> {after:.1f} dB")
    assert after > before + 3, "model failed to improve SNR by 3 dB"
    os.remove("demo.keras")
    print("demo ok")


def wavs(d):
    files = sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)
                   + glob.glob(os.path.join(d, "**", "*.flac"), recursive=True))
    if not files:
        raise SystemExit(f"No .wav or .flac files found under {d}")
    return [load(f) for f in files]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--clean", required=True); t.add_argument("--noise", required=True)
    t.add_argument("--out", default="cj1.keras")
    t.add_argument("--clips", type=int, default=2000); t.add_argument("--epochs", type=int, default=30)
    e = sub.add_parser("enhance")
    e.add_argument("input"); e.add_argument("output"); e.add_argument("--model", default="cj1.keras")
    sub.add_parser("demo")
    a = ap.parse_args()
    if a.cmd == "train":
        train(wavs(a.clean), wavs(a.noise), a.out, a.clips, a.epochs)
    elif a.cmd == "enhance":
        import keras
        sf.write(a.output, enhance(keras.models.load_model(a.model), load(a.input)), SR)
        print("wrote", a.output)
    else:
        demo()
