import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from scipy.signal import lfilter
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.diagnostic import (het_breuschpagan, het_white,
                                          acorr_breusch_godfrey, acorr_ljungbox)
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")
ALPHA = 0.05
SEED = 42
XCOLS = [f"X{i}" for i in range(1, 31)]


# ----------------------------------------------------------------------------
# Utilitas
# ----------------------------------------------------------------------------
def judul(teks):
    print("\n" + "=" * 78 + f"\n{teks}\n" + "=" * 78)


def keputusan(p, arti_tolak, arti_gagal):
    """Cetak keputusan uji pada alpha 5% beserta artinya."""
    if p < ALPHA:
        print(f"   -> p = {p:.4g} < {ALPHA}: TOLAK H0. {arti_tolak}")
    else:
        print(f"   -> p = {p:.4g} >= {ALPHA}: GAGAL TOLAK H0. {arti_gagal}")


def simpan(fig, outdir, nama):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, nama), dpi=130)
    plt.close(fig)


def make_demo(n, seed=SEED):
    """Data simulasi dengan struktur mirip data tugas:
    blok prediktor berkorelasi tinggi, varians galat tidak konstan, galat AR(1)."""
    rng = np.random.default_rng(seed)
    lat = rng.standard_normal((n, 9))
    cols = []
    for b in range(5):                      # X1..X25: 5 blok x 5 variabel
        for _ in range(5):
            cols.append(lat[:, b] + 0.25 * rng.standard_normal(n))
    for _ in range(2):
        cols.append(lat[:, 5] + 0.3 * rng.standard_normal(n))
    for _ in range(2):
        cols.append(lat[:, 6] + 0.3 * rng.standard_normal(n))
    cols.append(lat[:, 7])
    X = np.column_stack(cols)
    beta = rng.normal(0, 1, 30)
    sigma = np.exp(0.4 * X[:, 5])           # heteroskedastis
    u = rng.standard_normal(n) * sigma
    e = lfilter([1.0], [1.0, -0.5], u)      # AR(1) rho = 0.5
    y = 5 + X @ beta + e
    df = pd.DataFrame(X, columns=XCOLS)
    df["Y"] = y
    return df


# ----------------------------------------------------------------------------
# Ridge (closed form, CV lewat statistik cukup -> cepat untuk n besar)
# ----------------------------------------------------------------------------
def ridge_fit(Xtr, ytr, k_folds=5, lambdas=None):
    """Ridge: min RSS + lambda*||beta||^2 pada X terstandarisasi (intersep tidak dihukum).
    Lambda dipilih dengan k-fold CV (blok berurutan). Mengembalikan parameter pada skala
    asli [intersep, b1..b30], lambda, df efektif, dan info untuk grafik."""
    if lambdas is None:
        lambdas = np.logspace(-2, 9, 45)
    n, p = Xtr.shape
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Z = (Xtr - mu) / sd
    ymu = ytr.mean()
    yc = ytr - ymu

    edges = np.linspace(0, n, k_folds + 1, dtype=int)
    folds = []
    for a, b in zip(edges[:-1], edges[1:]):
        Zk, yk = Z[a:b], yc[a:b]
        folds.append((Zk.T @ Zk, Zk.T @ yk, yk @ yk))
    G = sum(f[0] for f in folds)
    c = sum(f[1] for f in folds)
    I = np.eye(p)

    cv, path = [], []
    for lam in lambdas:
        sse = 0.0
        for Gk, ck, yyk in folds:
            beta = np.linalg.solve(G - Gk + lam * I, c - ck)
            sse += yyk - 2 * beta @ ck + beta @ Gk @ beta   # SSE fold validasi
        cv.append(np.sqrt(sse / n))
        path.append(np.linalg.solve(G + lam * I, c))
    cv, path = np.array(cv), np.array(path)
    best = int(np.argmin(cv))
    lam = lambdas[best]

    beta_std = path[best]
    beta = beta_std / sd
    intercept = ymu - mu @ beta
    ev = np.linalg.eigvalsh(G)
    df_eff = 1.0 + np.sum(ev / (ev + lam))          # trace(H) + intersep
    return dict(params=np.r_[intercept, beta], lam=lam, df=df_eff,
                lambdas=lambdas, cv=cv, path=path / sd, best=best)


# ----------------------------------------------------------------------------
# Evaluasi model
# ----------------------------------------------------------------------------
def evaluasi(nama, params, Xtr, ytr, Xte, yte, k_eff, aic_native):
    n = len(ytr)
    rss = np.sum((ytr - Xtr @ params) ** 2)
    rmse_tr = np.sqrt(rss / n)
    rmse_te = np.sqrt(np.mean((yte - Xte @ params) ** 2))
    # AIC "basis sama": likelihood Gaussian homoskedastis pada residual skala asli
    aic_common = n * (np.log(2 * np.pi) + np.log(rss / n) + 1) + 2 * k_eff
    return dict(Model=nama, RMSE_train=rmse_tr, RMSE_test=rmse_te,
                AIC_native=aic_native, AIC_common=aic_common, df_efektif=k_eff)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="raw_data_simulasi.txt")
    ap.add_argument("--sep", default="\t")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--demo", type=int, default=0, help="pakai data simulasi n baris")
    ap.add_argument("--split", choices=["sequential", "random"], default="sequential")
    ap.add_argument("--outdir", default="output")
    args, _ = ap.parse_known_args()   # abaikan argumen tambahan dari Colab/Jupyter (-f ...)
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    # ------------------------------------------------------------------ 1. DATA
    judul("1. MEMUAT DATA & EKSPLORASI")
    df = make_demo(args.demo) if args.demo else pd.read_csv(
        args.data, sep=args.sep, nrows=args.nrows)
    print(f"Dimensi data : {df.shape}  | missing value: {int(df.isna().sum().sum())}")
    df = df.dropna()
    X_all = df[XCOLS].to_numpy(float)
    y_all = df["Y"].to_numpy(float)
    n = len(y_all)
    print("Ringkasan Y  :", {k: round(v, 4) for k, v in
                              df["Y"].describe()[["mean", "std", "min", "max"]].items()})

    # Split: sequential menjaga urutan baris (penting untuk autokorelasi/GLS)
    n_tr = int(0.8 * n)
    idx = np.arange(n) if args.split == "sequential" else rng.permutation(n)
    tr, te = idx[:n_tr], idx[n_tr:]
    Xtr_raw, ytr, Xte_raw, yte = X_all[tr], y_all[tr], X_all[te], y_all[te]
    Xtr = sm.add_constant(Xtr_raw, has_constant="add")
    Xte = sm.add_constant(Xte_raw, has_constant="add")
    names = ["const"] + XCOLS
    print(f"Data latih   : {len(ytr):,} baris | data uji: {len(yte):,} baris ({args.split})")

    # ------------------------------------------------------------------ 2. OLS AWAL
    judul("2. MODEL OLS AWAL (dasar untuk seluruh uji asumsi)")
    ols = sm.OLS(ytr, Xtr).fit()
    resid, fit = ols.resid, ols.fittedvalues
    print(f"R2 = {ols.rsquared:.4f} | Adj R2 = {ols.rsquared_adj:.4f} | "
          f"F-stat p-value = {ols.f_pvalue:.3g}")

    # ------------------------------------------------------------------ 3. NORMALITAS
    judul("3. UJI NORMALITAS SISAAN   (H0: sisaan berdistribusi normal)")
    jb, jb_p, skew, kurt = jarque_bera(resid)
    print(f"Skewness = {skew:.4f} | Kurtosis = {kurt:.4f} (normal: 0 dan 3)")
    print(f"[Jarque-Bera] JB = {jb:.2f}")
    keputusan(jb_p, "Sisaan tidak normal.", "Sisaan konsisten dengan normal.")
    k2, p_dag = stats.normaltest(resid)
    print(f"[D'Agostino-Pearson] K2 = {k2:.2f}")
    keputusan(p_dag, "Sisaan tidak normal.", "Sisaan konsisten dengan normal.")
    sub = rng.choice(resid, size=min(5000, len(resid)), replace=False)
    sw, p_sw = stats.shapiro(sub)
    print(f"[Shapiro-Wilk pada sampel acak 5000] W = {sw:.4f}")
    keputusan(p_sw, "Sisaan tidak normal.", "Sisaan konsisten dengan normal.")
    print("   Catatan: n sangat besar membuat uji formal sangat sensitif (penyimpangan kecil "
          "pun ditolak). Nilai skewness/kurtosis dan QQ-plot lebih informatif; "
          "dengan n besar, inferensi koefisien tetap valid secara asimtotik (CLT).")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(resid, bins=100, density=True, alpha=0.7)
    xs = np.linspace(resid.min(), resid.max(), 300)
    ax[0].plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std()), "r")
    ax[0].set_title("Histogram sisaan vs kurva normal")
    sm.qqplot(rng.choice(resid, size=min(20000, len(resid)), replace=False),
              line="s", ax=ax[1], markersize=2)
    ax[1].set_title("QQ-plot sisaan")
    simpan(fig, args.outdir, "01_normalitas.png")

    # ------------------------------------------------------------------ 4. MULTIKOLINEARITAS
    judul("4. UJI MULTIKOLINEARITAS   (VIF > 10 = serius, > 5 = perlu perhatian)")
    corr = np.corrcoef(Xtr_raw, rowvar=False)
    vif = np.diag(np.linalg.pinv(corr))             # VIF_j = diag(R^-1)
    vif_df = pd.DataFrame({"Variabel": XCOLS, "VIF": vif}).sort_values("VIF", ascending=False)
    print(vif_df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"Jumlah variabel VIF > 10: {(vif > 10).sum()} dari 30 | VIF > 5: {(vif > 5).sum()}")
    ev_corr = np.linalg.eigvalsh(corr)
    print(f"Condition number (X terstandarisasi) = {np.sqrt(ev_corr.max() / ev_corr.min()):.1f} "
          "(> 30 indikasi multikolinearitas kuat)")
    vif_df.to_csv(os.path.join(args.outdir, "vif.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(30)); ax.set_xticklabels(XCOLS, rotation=90, fontsize=7)
    ax.set_yticks(range(30)); ax.set_yticklabels(XCOLS, fontsize=7)
    fig.colorbar(im); ax.set_title("Matriks korelasi antar prediktor")
    simpan(fig, args.outdir, "02_korelasi_vif.png")

    # ------------------------------------------------------------------ 5. HETEROSKEDASTISITAS
    judul("5. UJI HETEROSKEDASTISITAS   (H0: varians sisaan konstan / homoskedastis)")
    lm, lm_p, _, _ = het_breuschpagan(resid, Xtr)
    print(f"[Breusch-Pagan] LM = {lm:.2f}")
    keputusan(lm_p, "Varians sisaan tidak konstan -> OLS tidak efisien, SE bias; perlu WLS/robust.",
              "Tidak ada bukti heteroskedastisitas.")
    Z = sm.add_constant(np.column_stack([fit, fit ** 2]))
    aux = sm.OLS(resid ** 2, Z).fit()
    lm_w = len(resid) * aux.rsquared
    p_w = stats.chi2.sf(lm_w, 2)
    print(f"[White versi ringkas: e^2 ~ y_hat + y_hat^2, data penuh] LM = {lm_w:.2f}")
    keputusan(p_w, "Varians sisaan bergantung pada nilai prediksi.", "Tidak ada bukti heteroskedastisitas.")
    try:
        m = min(20000, len(resid))
        ii = rng.choice(len(resid), m, replace=False)
        lm_f, p_f, _, _ = het_white(resid[ii], Xtr[ii])
        print(f"[White penuh (kuadrat + interaksi), sampel {m:,}] LM = {lm_f:.2f}")
        keputusan(p_f, "Terdapat heteroskedastisitas.", "Tidak ada bukti heteroskedastisitas.")
    except Exception as ex:
        print("   White penuh dilewati:", ex)
    ii = rng.choice(len(resid), min(5000, len(resid)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(fit[ii], resid[ii], s=3, alpha=0.4)
    ax.axhline(0, color="r"); ax.set_xlabel("Nilai prediksi"); ax.set_ylabel("Sisaan")
    ax.set_title("Sisaan vs prediksi (pola corong = heteroskedastis)")
    simpan(fig, args.outdir, "03_heteroskedastisitas.png")

    # ------------------------------------------------------------------ 6. AUTOKORELASI
    judul("6. UJI AUTOKORELASI SISAAN   (H0: tidak ada autokorelasi; urutan baris = urutan data)")
    dw = durbin_watson(resid)
    rho1 = acf(resid, nlags=1, fft=True)[1]
    print(f"[Durbin-Watson] DW = {dw:.4f} (~2 = tidak ada; <2 positif; >2 negatif) | rho lag-1 = {rho1:.4f}")
    bg_lm, bg_p, _, _ = acorr_breusch_godfrey(ols, nlags=5)
    print(f"[Breusch-Godfrey, 5 lag] LM = {bg_lm:.2f}")
    keputusan(bg_p, "Ada autokorelasi sisaan -> SE OLS bias; perlu GLS/AR.", "Tidak ada bukti autokorelasi.")
    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    print(f"[Ljung-Box, 10 lag] Q = {lb['lb_stat'].iloc[0]:.2f}")
    keputusan(float(lb["lb_pvalue"].iloc[0]), "Ada autokorelasi.", "Tidak ada bukti autokorelasi.")
    ac = acf(resid, nlags=30, fft=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, 31), ac[1:])
    bound = 1.96 / np.sqrt(len(resid))
    ax.axhline(bound, ls="--", color="r"); ax.axhline(-bound, ls="--", color="r")
    ax.set_xlabel("Lag"); ax.set_title("ACF sisaan OLS")
    simpan(fig, args.outdir, "04_acf_sisaan.png")

    # ------------------------------------------------------------------ 7. MODEL
    judul("7. PEMODELAN")
    hasil, koef = [], {"variabel": names}

    # --- OLS
    koef["OLS"] = ols.params
    hasil.append(evaluasi("OLS", ols.params, Xtr, ytr, Xte, yte, Xtr.shape[1], ols.aic))
    print("[OLS] selesai.")

    # --- WLS: bobot = 1 / varians terestimasi (model varians multiplikatif ala Harvey)
    #     log(e^2) ~ X  -> varians_hat = exp(fitted). Konstanta skala tidak memengaruhi bobot relatif.
    aux_v = sm.OLS(np.log(resid ** 2 + 1e-12), Xtr).fit()
    w = 1.0 / np.exp(aux_v.fittedvalues)
    wls = sm.WLS(ytr, Xtr, weights=w).fit()
    koef["WLS"] = wls.params
    hasil.append(evaluasi("WLS", wls.params, Xtr, ytr, Xte, yte, Xtr.shape[1], wls.aic))
    print("[WLS] selesai (bobot dari regresi log(sisaan^2) pada X).")

    # --- GLS: FGLS dengan galat AR(1) (Cochrane-Orcutt iteratif). Matriks Omega penuh
    #     1,2 juta x 1,2 juta tidak mungkin dibentuk -> struktur AR(1) dipakai.
    gls_m = sm.GLSAR(ytr, Xtr, rho=1)
    gls = gls_m.iterative_fit(maxiter=10)
    rho_hat = float(np.atleast_1d(gls_m.rho)[0])
    koef["GLS"] = gls.params
    hasil.append(evaluasi("GLS (AR1)", gls.params, Xtr, ytr, Xte, yte, Xtr.shape[1] + 1, gls.aic))
    print(f"[GLS] selesai. rho AR(1) terestimasi = {rho_hat:.4f} "
          "(rho ~ 0 berarti GLS ~ OLS)")

    # --- Ridge
    rd = ridge_fit(Xtr_raw, ytr)
    koef["Ridge"] = rd["params"]
    k_r = rd["df"]
    rss_r = np.sum((ytr - Xtr @ rd["params"]) ** 2)
    aic_r = len(ytr) * (np.log(2 * np.pi) + np.log(rss_r / len(ytr)) + 1) + 2 * k_r
    hasil.append(evaluasi("Ridge", rd["params"], Xtr, ytr, Xte, yte, k_r, aic_r))
    print(f"[Ridge] lambda optimal (5-fold CV) = {rd['lam']:.4g} | df efektif = {k_r:.2f} dari 31")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogx(rd["lambdas"], rd["cv"]); ax[0].axvline(rd["lam"], color="r", ls="--")
    ax[0].set_xlabel("lambda"); ax[0].set_ylabel("RMSE CV"); ax[0].set_title("Pemilihan lambda (CV)")
    ax[1].semilogx(rd["lambdas"], rd["path"]); ax[1].axvline(rd["lam"], color="r", ls="--")
    ax[1].set_xlabel("lambda"); ax[1].set_ylabel("koefisien"); ax[1].set_title("Ridge path")
    simpan(fig, args.outdir, "05_ridge.png")

    # ------------------------------------------------------------------ 8. PERBANDINGAN
    judul("8. PERBANDINGAN MODEL (RMSE & AIC)")
    tab = pd.DataFrame(hasil).set_index("Model")
    print(tab.to_string(float_format=lambda v: f"{v:,.4f}"))
    print("\nModel terbaik menurut RMSE data uji :", tab["RMSE_test"].idxmin())
    print("Model terbaik menurut AIC (native)   :", tab["AIC_native"].idxmin(),
          "  <- hati-hati: AIC WLS/GLS dihitung pada data tertransformasi")
    print("Model terbaik menurut AIC (basis sama):", tab["AIC_common"].idxmin(),
          "  <- perbandingan yang adil (likelihood Gaussian pada residual skala asli)")
    print("\nCatatan interpretasi:\n"
          " - RMSE dihitung pada data uji dengan skala Y asli -> langsung sebanding antar model.\n"
          " - AIC_native untuk WLS/GLS memakai likelihood data terboboti/tertransformasi, jadi tidak\n"
          "   sepenuhnya sebanding dengan OLS; AIC_common disediakan sebagai pembanding.\n"
          " - Karena n sangat besar, varians akibat multikolinearitas kecil; Ridge cenderung mirip OLS\n"
          "   (lambda optimal kecil). Manfaat WLS/GLS terutama pada ketepatan galat baku/inferensi,\n"
          "   bukan pada RMSE prediksi.")
    tab.to_csv(os.path.join(args.outdir, "model_comparison.csv"))
    pd.DataFrame(koef).to_csv(os.path.join(args.outdir, "coefficients.csv"), index=False)
    print(f"\nSelesai dalam {time.time() - t0:.1f} detik. Hasil tersimpan di folder: {args.outdir}/")


if __name__ == "__main__":
    main()