import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from lazypredict.Supervised import LazyClassifier


# Veri Yükleme 
COLS = [
    "Severity", "Source",
    "Start_Lat", "Start_Lng", "City", "County", "State", "Zipcode", "Timezone",
    "Start_Time", "End_Time",
    "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
    "Weather_Condition", "Wind_Direction",
    "Distance(mi)", "Street", "Description",
    "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction",
    "No_Exit", "Railway", "Roundabout", "Station", "Stop",
    "Traffic_Calming", "Traffic_Signal", "Turning_Loop",
]
DTYPES = {
    "Severity": "int8",
    "City": "category", "County": "category", "State": "category",
    "Timezone": "category", "Weather_Condition": "category",
    "Wind_Direction": "category",
    "Sunrise_Sunset": "category", "Civil_Twilight": "category",
    "Nautical_Twilight": "category", "Astronomical_Twilight": "category",
    "Amenity": "bool", "Bump": "bool", "Crossing": "bool",
    "Give_Way": "bool", "Junction": "bool", "No_Exit": "bool",
    "Railway": "bool", "Roundabout": "bool", "Station": "bool",
    "Stop": "bool", "Traffic_Calming": "bool", "Traffic_Signal": "bool",
    "Turning_Loop": "bool",
}

chunks = []
for chunk in pd.read_csv(
    "US_Accidents_March23.csv",
    usecols=COLS,
    dtype=DTYPES,
    chunksize=200_000,
):
    sampled = chunk.groupby("Severity", group_keys=False).sample(
        frac=0.15, random_state=42
    )
    chunks.append(sampled)

df = pd.concat(chunks, ignore_index=True)
print(f"Shape: {df.shape}")


# Tarih Parse
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
df["End_Time"]   = pd.to_datetime(df["End_Time"],   errors="coerce")

# Feature Engineering 
df["duration_min"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
df["duration_min"] = df["duration_min"].clip(lower=0, upper=600)  # burada temizle
df["hour"]         = df["Start_Time"].dt.hour
df["day_of_week"]  = df["Start_Time"].dt.dayofweek
df["month"]        = df["Start_Time"].dt.month
df["year"]         = df["Start_Time"].dt.year
df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype("int8")
df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype("int8")
df["low_visibility"] = (df["Visibility(mi)"] < 1).astype("int8")
df["heavy_precip"]   = (df["Precipitation(in)"] > 0.3).astype("int8")

poi_cols = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
    "Railway", "Roundabout", "Station", "Stop",
    "Traffic_Calming", "Traffic_Signal", "Turning_Loop",
]
df["poi_count"]   = df[poi_cols].sum(axis=1).astype("int8")
df["desc_length"] = df["Description"].str.len().fillna(0).astype("int16")

df.drop(columns=["Start_Time", "End_Time", "Description", "Street"], inplace=True)
print(f"Final shape: {df.shape}")

df["Source"] = df["Source"].map({
    "Source1": "MapQuest",
    "Source2": "Bing",
    "Source3": "Other"
})

df = df[df["Source"].isin(["Bing", "MapQuest"])]

# Parquet: Ham veriyi encode ETMEDEN kaydet 
df.to_parquet("accidents_sample.parquet", index=False)
print("Parquet kaydedildi.")


# EDA — KEŞİFSEL VERİ ANALİZİ
print(df["Source"].unique())
print(df["Source"].isna().sum())
print("\n── EDA Başlıyor ──")

# 1. Severity Dağılımı
plt.figure(figsize=(7, 4))
ax = sns.countplot(data=df, x="Severity", hue="Severity", legend=False,
                   palette={1:"#2196F3", 2:"#FF9800", 3:"#F44336", 4:"#7B1FA2"})
ax.bar_label(ax.containers[0], fmt="%d")
plt.title("Kaza Şiddeti Dağılımı (Severity)")
plt.xlabel("Severity (1=hafif, 4=ağır)")
plt.ylabel("Kaza Sayısı")
plt.tight_layout()
plt.savefig("eda_severity.png", dpi=150)
plt.show()

# 2. Saate göre kaza yoğunluğu (sev_group ile okunabilir)
df["sev_group"] = df["Severity"].map({1:"Hafif(1-2)", 2:"Hafif(1-2)", 3:"Ağır(3-4)", 4:"Ağır(3-4)"})
plt.figure(figsize=(12, 4))
sns.countplot(data=df, x="hour", hue="sev_group",
              palette={"Hafif(1-2)":"#2196F3", "Ağır(3-4)":"#F44336"})
plt.title("Saate Göre Kaza Dağılımı (Hafif vs Ağır)")
plt.xlabel("Saat")
plt.ylabel("Kaza Sayısı")
plt.tight_layout()
plt.savefig("eda_hour.png", dpi=150)
plt.show()

# 3. Güne göre kaza
gun_isimleri = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
gun_counts = df["day_of_week"].value_counts().sort_index()
gun_df = pd.DataFrame({"gun": gun_isimleri, "count": gun_counts.values})
plt.figure(figsize=(8, 4))
ax = sns.barplot(data=gun_df, x="gun", y="count", hue="gun", legend=False,
                 palette=["#1565C0","#1976D2","#1E88E5","#2196F3","#42A5F5","#90CAF9","#BBDEFB"])
plt.title("Güne Göre Kaza Sayısı")
plt.ylabel("Kaza Sayısı")
plt.tight_layout()
plt.savefig("eda_dayofweek.png", dpi=150)
plt.show()

# 4. Aya göre kaza trendi
yil_renkleri = ["#E53935","#8E24AA","#1E88E5","#00ACC1","#43A047","#FB8C00","#6D4C41","#546E7A"]
plt.figure(figsize=(10, 4))
ay_counts = df.groupby(["year", "month"]).size().reset_index(name="count")
for i, yil in enumerate(sorted(ay_counts["year"].unique())):
    subset = ay_counts[ay_counts["year"] == yil]
    plt.plot(subset["month"], subset["count"], marker="o",
             color=yil_renkleri[i % len(yil_renkleri)], linewidth=2, label=str(yil))
plt.title("Yıl/Ay Bazında Kaza Trendi")
plt.xlabel("Ay")
plt.ylabel("Kaza Sayısı")
plt.legend(title="Yıl", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig("eda_trend.png", dpi=150)
plt.show()

# 5. En çok kaza olan 15 eyalet
top_states = df["State"].value_counts().head(15).reset_index()
top_states.columns = ["State", "count"]
plt.figure(figsize=(10, 5))
sns.barplot(data=top_states, x="count", y="State", hue="State", legend=False,
            palette=[plt.cm.RdYlBu_r(i / 15) for i in range(15)])
plt.title("En Fazla Kaza Olan 15 Eyalet")
plt.xlabel("Kaza Sayısı")
plt.tight_layout()
plt.savefig("eda_states.png", dpi=150)
plt.show()

# 6. Korelasyon ısı haritası
num_cols_eda = [
    "Severity", "Temperature(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
    "duration_min", "poi_count", "hour", "is_rush_hour", "is_weekend"
]
plt.figure(figsize=(12, 9))
corr = df[num_cols_eda].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
            linewidths=0.5, annot_kws={"size": 9},
            vmin=-1, vmax=1)
plt.title("Değişkenler Arası Korelasyon")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150)
plt.show()

# 7. Gece/gündüz kaza dağılımı
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sunrise_Sunset", hue="Severity",
              palette=["#2196F3","#FF9800","#F44336","#7B1FA2"])
plt.title("Gece / Gündüz Kaza Şiddeti")
plt.xlabel("")
plt.tight_layout()
plt.savefig("eda_daynight.png", dpi=150)
plt.show()

# 8. Coğrafi Harita
plt.figure(figsize=(15, 9))
colors_map = {1: '#43A047', 2: '#FF9800', 3: '#E53935', 4: '#5E35B1'}
sizes_map  = {1: 30, 2: 1, 3: 10, 4: 5}
for sev in [1, 2, 3, 4]:
    subset = df[df['Severity'] == sev]
    if not subset.empty:
        plt.scatter(subset['Start_Lng'], subset['Start_Lat'],
                    c=colors_map[sev], s=sizes_map[sev],
                    label=f'Severity {sev}', alpha=0.5)
plt.xlim([-125, -66])
plt.ylim([24, 50])
plt.title('US Accidents — Coğrafi Dağılım (Severity)', size=16, pad=20)
plt.xlabel('Boylam')
plt.ylabel('Enlem')
plt.legend(markerscale=3, loc='lower right')
plt.grid(True, linestyle='--', alpha=0.2)
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig("eda_map_final.png", dpi=150)
plt.show()

# ── 9. Kaynağa Göre Severity ──

plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Severity", hue="Source",
              palette={"Bing":"#1E88E5", "MapQuest":"#E53935", "Other":"#43A047"})
plt.title("Kaynağa Göre Severity Dağılımı (Bing vs MapQuest)")
plt.xlabel("Severity (1=hafif, 4=ağır)")
plt.ylabel("Kaza Sayısı")
plt.tight_layout()
plt.savefig("eda_source_severity.png", dpi=150)
plt.show()

# --- Kaynak Bazlı Korelasyon Analizi ---
for name in ["Bing", "MapQuest"]:
    source_df = df[df["Source"] == name]
    
    corr_cols = [
        "Severity", "Temperature(F)", "Humidity(%)", "Visibility(mi)",
        "Wind_Speed(mph)", "duration_min", "poi_count", "hour", "is_weekend"
    ]
    
    source_corr = source_df[corr_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(source_corr, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, vmin=-1, vmax=1,
                linewidths=0.5, annot_kws={"size": 10})
    plt.title(f"Korelasyon Matrisi — Kaynak: {name}")
    plt.tight_layout()
    plt.savefig(f"corr_{name.lower()}.png", dpi=150)
    plt.show()
print("EDA grafikleri kaydedildi.")

# sev_group sadece EDA için kullanıldı, modele verme
df.drop(columns=["sev_group"], inplace=True, errors="ignore")
# Kategorikleri encode et (EDA sonrası, parquet kaydından sonra)

cat_cols = [
    "City", "County", "State", "Timezone",
    "Weather_Condition", "Wind_Direction",
    "Sunrise_Sunset", "Civil_Twilight",
    "Nautical_Twilight", "Astronomical_Twilight",
]
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))

df["Zipcode"] = pd.to_numeric(
    df["Zipcode"].astype(str).str[:5], errors="coerce"
).fillna(-1).astype("int32")

num_cols = [
    "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "duration_min",
]
df[num_cols] = df[num_cols].fillna(df[num_cols].median())


# ── Kaynağa göre ayır ──
df_bing     = df[df["Source"] == "Bing"].copy().drop(columns=["Source"])
df_mapquest = df[df["Source"] == "MapQuest"].copy().drop(columns=["Source"])
print(f"Bing: {df_bing.shape}, MapQuest: {df_mapquest.shape}")


def lazy_compare(data, source_name):
    X = data.drop(columns=["Severity"])
    y = data["Severity"] - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # LazyPredict büyük veride yavaş — örnekle
    X_train_s = X_train.sample(n=10000, random_state=42)
    y_train_s = y_train.loc[X_train_s.index]
    X_test_s  = X_test.sample(n=3000, random_state=42)
    y_test_s  = y_test.loc[X_test_s.index]

    print(f"\nLazyPredict çalışıyor — {source_name}...")
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models_df, predictions = clf.fit(X_train_s, X_test_s, y_train_s, y_test_s)

    print(f"\n── {source_name} — LazyPredict Sonuçları (Top 15) ──")
    print(models_df.head(15))

    # Grafik
    plt.figure(figsize=(10, 7))
    top15 = models_df.head(15)
    colors = ["#E53935" if i == 0 else "#1E88E5" for i in range(len(top15))]
    top15["Accuracy"].sort_values().plot(kind="barh", color=colors[::-1])
    plt.title(f"LazyPredict Model Karşılaştırması — {source_name}")
    plt.xlabel("Accuracy")
    plt.axvline(x=top15["Accuracy"].mean(), color="gray",
                linestyle="--", label="Ortalama")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"lazy_{source_name.lower()}.png", dpi=150)
    plt.show()

    return models_df

lazy_bing     = lazy_compare(df_bing,     "Bing")
lazy_mapquest = lazy_compare(df_mapquest, "MapQuest")

# İki kaynağı karşılaştır
print("\n── Ortak En İyi Modeller ──")
common = pd.DataFrame({
    "Bing_Acc":     lazy_bing["Accuracy"],
    "MapQuest_Acc": lazy_mapquest["Accuracy"],
}).dropna()
common["Ortalama"] = common.mean(axis=1)
print(common.sort_values("Ortalama", ascending=False).head(10))


#---------------------------------------------------------------------------------------------
# ── Model eğitim fonksiyonu ──
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time

def train_and_compare(data, source_name):
    X = data.drop(columns=["Severity"])
    y = data["Severity"] - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # Ayrıca validation seti ayır (train'den %20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = dict(zip(np.unique(y_train), weights))
    sw = np.array([cw[t] for t in y_train])

    models = {
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            tree_method="hist", eval_metric="mlogloss", random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=42, verbose=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=12,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
    }

    results = {}
    trained_models = {}

    print(f"\n{'='*50}")
    print(f"  KAYNAK: {source_name}")
    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"{'='*50}")

    for model_name, model in models.items():
        print(f"\n── {model_name} eğitiliyor...")
        start = time.time()

        if model_name == "XGBoost":
            model.fit(X_train, y_train, sample_weight=sw,
                      eval_set=[(X_val, y_val)], verbose=False)
        elif model_name == "LightGBM":
            model.fit(X_train, y_train, sample_weight=sw,
                      eval_set=[(X_val, y_val)])
        else:
            model.fit(X_train, y_train, sample_weight=sw)

        elapsed = time.time() - start

        # Validation skoru
        y_val_pred = model.predict(X_val)
        val_report = classification_report(y_val, y_val_pred,
                                           target_names=["S1","S2","S3","S4"],
                                           output_dict=True)

        # Test skoru
        y_test_pred = model.predict(X_test)
        test_report = classification_report(y_test, y_test_pred,
                                            target_names=["S1","S2","S3","S4"],
                                            output_dict=True)

        results[model_name] = {
            "val_f1":  val_report["weighted avg"]["f1-score"],
            "test_f1": test_report["weighted avg"]["f1-score"],
            "time":    elapsed,
        }
        trained_models[model_name] = (model, X_test, y_test, y_test_pred)

        print(f"   Süre: {elapsed:.1f}s | Val F1: {val_report['weighted avg']['f1-score']:.3f} | Test F1: {test_report['weighted avg']['f1-score']:.3f}")

    # Karşılaştırma tablosu
    print(f"\n── {source_name} Model Karşılaştırması ──")
    results_df = pd.DataFrame(results).T
    print(results_df.round(3))

    # Karşılaştırma grafiği
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(results))
    bars1 = ax.bar(x - 0.2, results_df["val_f1"],  0.35, label="Val F1",  color="#1E88E5")
    bars2 = ax.bar(x + 0.2, results_df["test_f1"], 0.35, label="Test F1", color="#E53935")
    ax.bar_label(bars1, fmt="%.3f", fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index)
    ax.set_ylim(0, 1)
    ax.set_title(f"Model Karşılaştırması — {source_name}")
    ax.set_ylabel("Weighted F1 Score")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"model_compare_{source_name.lower()}.png", dpi=150)
    plt.show()

    # En iyi modelin confusion matrix
    best_name = max(results, key=lambda k: results[k]["test_f1"])
    best_model, X_te, y_te, y_pred = trained_models[best_name]
    print(f"\n En iyi model: {best_name} (Test F1: {results[best_name]['test_f1']:.3f})")
    print(classification_report(y_te, y_pred, target_names=["S1","S2","S3","S4"]))

    plt.figure(figsize=(7, 5))
    sns.heatmap(confusion_matrix(y_te, y_pred), annot=True, fmt="d",
                cmap="Blues", xticklabels=["S1","S2","S3","S4"],
                yticklabels=["S1","S2","S3","S4"])
    plt.title(f"Confusion Matrix — {source_name} / {best_name}")
    plt.tight_layout()
    plt.savefig(f"model_best_cm_{source_name.lower()}.png", dpi=150)
    plt.show()

    return best_model, X_test, y_test

# ── Her kaynağı eğit ve karşılaştır ──
best_bing,     X_test_bing,     y_test_bing     = train_and_compare(df_bing,     "Bing")
best_mapquest, X_test_mapquest, y_test_mapquest = train_and_compare(df_mapquest, "MapQuest")

# ── En iyi modellerle Soft Voting Ensemble ──
X_all = pd.concat([X_test_bing, X_test_mapquest])
y_all = pd.concat([y_test_bing, y_test_mapquest])

prob_ensemble = (
    best_bing.predict_proba(X_all) +
    best_mapquest.predict_proba(X_all)
) / 2
y_pred_ensemble = np.argmax(prob_ensemble, axis=1)

print("\n══════════════════════════════════")
print("   ENSEMBLE MODEL PERFORMANSI     ")
print("══════════════════════════════════")
print(classification_report(y_all, y_pred_ensemble, target_names=["S1","S2","S3","S4"]))

plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_all, y_pred_ensemble), annot=True, fmt="d",
            cmap="Purples", xticklabels=["S1","S2","S3","S4"],
            yticklabels=["S1","S2","S3","S4"])
plt.title("Confusion Matrix — Ensemble (En İyi Bing + En İyi MapQuest)")
plt.tight_layout()
plt.savefig("model_cm_ensemble_best.png", dpi=150)
plt.show()

# ── Feature Importance karşılaştırması ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, model, name in zip(axes, [best_bing, best_mapquest], ["Bing", "MapQuest"]):
    if hasattr(model, "feature_importances_"):
        feat_imp = pd.Series(model.feature_importances_, index=X_test_bing.columns)
        feat_imp.nlargest(15).sort_values().plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"Feature Importance — {name}")
plt.tight_layout()
plt.savefig("model_feature_importance_best.png", dpi=150)
plt.show()

# Modeller

# Kaynak Bazlı Performans Ölçümü yapalımmmmm



import json

# ── DASHBOARD İÇİN SONUÇLARI KAYDET ──

# 1. Severity dağılımı
sev_counts = df["Severity"].value_counts().sort_index()
total = sev_counts.sum()

# 2. Saat dağılımı
hour_counts = df["hour"].value_counts().sort_index().reindex(range(24), fill_value=0)

# 3. Gün dağılımı
day_counts = df["day_of_week"].value_counts().sort_index().reindex(range(7), fill_value=0)

# 4. Yıl/Ay trendi
trend = df.groupby(["year","month"]).size().reset_index(name="count")
trend_dict = {}
for _, row in trend.iterrows():
    y = str(int(row["year"]))
    m = int(row["month"])
    if y not in trend_dict:
        trend_dict[y] = [None]*12
    trend_dict[y][m-1] = int(row["count"])

# 5. Top 15 eyalet
top_states = df["State"].value_counts().head(15)

# 6. Hava durumu
weather_counts = df["Weather_Condition"].value_counts().head(10)
weather_total  = weather_counts.sum()

# 7. Sıcaklık bins
temp_bins  = [-float("inf"),10,20,30,40,50,60,70,80,90,float("inf")]
temp_labels= ["<10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90",">90"]
temp_cut   = pd.cut(df["Temperature(F)"].dropna(), bins=temp_bins, labels=temp_labels)
temp_pct   = (temp_cut.value_counts(sort=False) / len(df) * 100).round(2)

# 8. Gece/gündüz
daynight = df["Sunrise_Sunset"].value_counts()

# 9. Korelasyon
num_cols_corr = [
    "Severity","Temperature(F)","Humidity(%)","Pressure(in)",
    "Visibility(mi)","Wind_Speed(mph)","Precipitation(in)",
    "duration_min","poi_count","hour","is_rush_hour","is_weekend"
]
corr_matrix = df[num_cols_corr].corr().round(3).values.tolist()

# 10. POI oranları
poi_cols = ["Amenity","Bump","Crossing","Give_Way","Junction","No_Exit",
            "Railway","Roundabout","Station","Stop",
            "Traffic_Calming","Traffic_Signal","Turning_Loop"]
poi_pct = (df[poi_cols].mean() * 100).round(2)

# 11. poi_count dağılımı
poi_count_dist = df["poi_count"].clip(upper=5).value_counts().sort_index()

# 12. Kaynak bazlı severity
bing_sev = df[df["Source"]=="Bing"]["Severity"].value_counts(normalize=True).sort_index()*100
mq_sev   = df[df["Source"]=="MapQuest"]["Severity"].value_counts(normalize=True).sort_index()*100
bing_n   = int(len(df[df["Source"]=="Bing"]))
mq_n     = int(len(df[df["Source"]=="MapQuest"]))

# 13. Lazy sonuçları
lazy_combined = pd.DataFrame({
    "Bing_Acc": lazy_bing["Accuracy"],
    "MQ_Acc":   lazy_mapquest["Accuracy"],
}).dropna()
lazy_combined["Avg"] = lazy_combined.mean(axis=1)
lazy_top = lazy_combined.sort_values("Avg", ascending=False).head(10)

# 14. Model sonuçları — train_and_compare fonksiyonunu modify et
# Fonksiyonun return'ünü genişlet:
def train_and_compare_v2(data, source_name):
    X = data.drop(columns=["Severity"])
    y = data["Severity"] - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = dict(zip(np.unique(y_train), weights))
    sw = np.array([cw[t] for t in y_train])

    models = {
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            tree_method="hist", eval_metric="mlogloss", random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=42, verbose=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=12,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
    }

    results      = {}
    trained_models = {}

    for model_name, model in models.items():
        start = time.time()
        if model_name == "XGBoost":
            model.fit(X_train, y_train, sample_weight=sw,
                      eval_set=[(X_val, y_val)], verbose=False)
        elif model_name == "LightGBM":
            model.fit(X_train, y_train, sample_weight=sw,
                      eval_set=[(X_val, y_val)])
        else:
            model.fit(X_train, y_train, sample_weight=sw)

        elapsed = time.time() - start

        y_val_pred  = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        val_rep  = classification_report(y_val,  y_val_pred,  output_dict=True)
        test_rep = classification_report(y_test, y_test_pred, output_dict=True)
        cm       = confusion_matrix(y_test, y_test_pred).tolist()

        # Feature importance
        feat_imp = {}
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=X.columns)
            feat_imp = fi.nlargest(15).round(4).to_dict()

        results[model_name] = {
            "val_f1":    round(val_rep["weighted avg"]["f1-score"], 4),
            "test_f1":   round(test_rep["weighted avg"]["f1-score"], 4),
            "time":      round(elapsed, 1),
            "cm":        cm,
            "feat_imp":  feat_imp,
            "class_report": {
                k: {m: round(v,4) for m,v in vals.items()}
                for k, vals in test_rep.items()
                if k in ["0","1","2","3"]
            }
        }
        trained_models[model_name] = (model, X_test, y_test, y_test_pred)
        print(f"  {model_name} | Val F1: {results[model_name]['val_f1']} | Test F1: {results[model_name]['test_f1']}")

    best_name = max(results, key=lambda k: results[k]["test_f1"])
    return trained_models[best_name][0], X_test, y_test, results

# Eğit
best_bing,     X_test_bing,     y_test_bing,     res_bing = train_and_compare_v2(df_bing,     "Bing")
best_mapquest, X_test_mapquest, y_test_mapquest, res_mq   = train_and_compare_v2(df_mapquest, "MapQuest")

# Ensemble
X_all = pd.concat([X_test_bing, X_test_mapquest])
y_all = pd.concat([y_test_bing, y_test_mapquest])
prob_ensemble   = (best_bing.predict_proba(X_all) + best_mapquest.predict_proba(X_all)) / 2
y_pred_ensemble = np.argmax(prob_ensemble, axis=1)
ens_rep = classification_report(y_all, y_pred_ensemble, output_dict=True)
ens_cm  = confusion_matrix(y_all, y_pred_ensemble).tolist()

# ── JSON'a yaz ──
results_data = {
    "meta": {
        "total_records":  int(total * (1/0.15)),
        "sample_size":    int(total),
        "bing_n":         bing_n,
        "mq_n":           mq_n,
    },
    "severity": {
        str(k): int(v) for k, v in sev_counts.items()
    },
    "hour":    hour_counts.tolist(),
    "day":     day_counts.tolist(),
    "trend":   trend_dict,
    "states": {
        "codes":  top_states.index.tolist(),
        "counts": top_states.tolist()
    },
    "weather": {
        "labels": weather_counts.index.tolist(),
        "pcts":   (weather_counts / weather_total * 100).round(2).tolist()
    },
    "temp": {
        "labels": temp_labels,
        "pcts":   temp_pct.tolist()
    },
    "daynight": {
        "Day":   float(daynight.get("Day", 0)),
        "Night": float(daynight.get("Night", 0))
    },
    "corr": {
        "labels": num_cols_corr,
        "matrix": corr_matrix
    },
    "poi": {
        "labels": poi_cols,
        "pcts":   poi_pct.tolist()
    },
    "poi_count": {
        "labels": [str(i) for i in poi_count_dist.index.tolist()],
        "counts": poi_count_dist.tolist()
    },
    "source_sev": {
        "bing": bing_sev.reindex([1,2,3,4], fill_value=0).round(2).tolist(),
        "mq":   mq_sev.reindex([1,2,3,4],  fill_value=0).round(2).tolist(),
    },
    "lazy": {
        "models": lazy_top.index.tolist(),
        "bing":   lazy_top["Bing_Acc"].round(4).tolist(),
        "mq":     lazy_top["MQ_Acc"].round(4).tolist(),
        "avg":    lazy_top["Avg"].round(4).tolist(),
    },
    "models": {
        "bing":     res_bing,
        "mapquest": res_mq,
    },
    "ensemble": {
        "accuracy":  round(ens_rep["accuracy"], 4),
        "f1":        round(ens_rep["weighted avg"]["f1-score"], 4),
        "precision": round(ens_rep["weighted avg"]["precision"], 4),
        "recall":    round(ens_rep["weighted avg"]["recall"], 4),
        "cm":        ens_cm,
    }
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results_data, f, ensure_ascii=False, indent=2)

print("\n✅ results.json kaydedildi — dashboard hazır!")












































