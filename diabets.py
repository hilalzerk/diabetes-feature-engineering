#############################################
# Miuul Bootcamp-DİYABET - FEATURE ENGINEERING PROJESİ
#############################################


# İş Problemi

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin
# edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli
# geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını
# gerçekleştirmeniz beklenmektedir.


# Veriseti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır.
# ABD'deki Arizona Eyaleti'nin enbüyük 5.şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde
# yapılan diyabet araştırması için kullanılan verilerdir.
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# =============================================================================
# VERİ SETİ SÜTUN AÇIKLAMALARI
# =============================================================================

# | Sütun Adı               | Türkçe Açıklama
# |-------------------------|------------------------------------------------------------------------
# | Pregnancies             | Hamilelik sayısı
# | Glucose                 | Glikoz
# | BloodPressure           | Kan Basıncı - Küçük tansiyon (mm Hg)
# | SkinThickness           | Cilt Kalınlığı
# | Insulin                 | İnsülin (mu U/ml)
# | DiabetesPedigreeFunction| Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon
# | BMI                     | Vücut Kitle Endeksi
# | Age                     | Yaş (yıl)
# | Outcome                 | Hastalığa sahip (1) ya da değil (0)
# =============================================================================

# Görev 1 : Keşifçi Veri Analizi
# Adım 1: Genel resmi inceleyiniz.
# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)
# Adım 5: Aykırı gözlem analizi yapınız.
# Adım 6: Eksik gözlem analizi yapınız.
# Adım 7: Korelasyon analizi yapınız.
# =============================================================================
# Görev 2 : Feature Engineering
# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.
# Adım 2: Yeni değişkenler oluşturunuz.
# Adım 3: Encoding işlemlerini gerçekleştiriniz.
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
# Adım 5: Model oluşturunuz.

# Gerekli kütüphane ve görsel ayarlar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Veri Setinin okunması
def load():
    data = pd.read_csv("Ödevler/Diabets/Data/diabetes.csv")
    return data

dataframe = load()

# =============================================================================
# Görev 1 : Keşifçi Veri Analizi
# =============================================================================
# Adım 1: Genel resmi inceleyiniz.

def check_df(dataframe, head=5):
    print("########## Sahepe ########")
    print(dataframe.shape)

    print("########## Types ########")
    print(dataframe.dtypes)

    print("########## Head ########")
    print(dataframe.head(head))

    print("########## Tail ########")
    print(dataframe.tail(head))

    print("########## NA ########")
    print(dataframe.isnull().sum())

    print("########## Quantiles ########")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(dataframe)

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

# Kategorik Değişken Analizi
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#####################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe, palette="viridis")
        plt.show(block=False)

for col in cat_cols:
     cat_summary(dataframe, col)


# Nümereik Değişken Analizi
def num_summary(dataframe,num_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    dataframe[num_cols].describe().T
    print(dataframe[num_cols].describe().T)

    if plot:
        dataframe[num_cols].hist()
        plt.xlabel(num_cols)
        plt.ylabel(num_cols)
        plt.show(block=True)

for col in num_cols:
    num_summary(dataframe,col, plot= False)


# # Adım 4: Hedef değişken analizi yapınız.
# (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

# Nümerik değişkene göre target analizi

def target_summary_with_num(dataframe, target, numerical_col):
    temp_df = dataframe.groupby(target).agg({numerical_col: "mean"})
    print(temp_df)
    print("##################################################")
    colors=["#cb4854", "#e49e7f"]
    temp_df.plot(kind="bar", y=numerical_col, color = colors)
    plt.show(block=True)

for col in num_cols:
    target_summary_with_num(dataframe, "Outcome", col)



# Adım 5: Aykırı gözlem analizi yapınız.

# IQR yöntemiyle alt ve üst eşik değerlerini hesaplama
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer var mı kontrolü - True/False döner
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Aykırı değerlerin kendisine erişim
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


print("HER SÜTUN İÇİN AYKIRI DEĞER VAR MI?")
for col in num_cols:
    low, up = outlier_thresholds(dataframe, col)
    print(f"  {col:30s} | Aykırı: {check_outlier(dataframe, col)} | Alt: {low:.2f} | Üst: {up:.2f}")



# Adım 6: Eksik gözlem analizi yapınız.

# Veri setinde ilk baktığımızda boş değer görünmesede bazı değerlerde 0 olamayacağını analiz etmiştik.
zero_columns = [col for col in dataframe.columns if (dataframe[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

zero_columns


# Adım 7: Korelasyon analizi yapınız.

# Korelasyon analizi
dataframe.corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(dataframe.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Feature Engineering Öncesi Correlation Matrix", fontsize=20)
plt.show(block=False)

# Base Model Kurulumu

y = dataframe["Outcome"] # bağımlı değişken - target
X = dataframe.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# importance tablosu
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num], palette="viridis")
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X)


# =============================================================================
# Görev 2 : Feature Engineering
# =============================================================================

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.


# nan değeri atanması

for col in zero_columns:
    dataframe[col] = np.where(dataframe[col] == 0, np.nan, dataframe[col])

# NaN değeri atandıktan sonra eksik gözlem analizi
dataframe.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(dataframe, na_name=True)



### EKSİK DEĞERLERİN DOLDURULMASI

for col in zero_columns:
    dataframe[col] = dataframe.groupby("Outcome")[col].transform(
        lambda x: x.fillna(x.median())
    )
    print(f"  {col:20s} → Outcome grubuna göre Medyan ile dolduruldu")

# Kalan eksikler varsa → global medyanla doldur
for col in zero_columns:
    dataframe[col] = dataframe[col].fillna(dataframe[col].median())

print("\n>> Doldurma sonrası eksik değer kontrolü:")
print(dataframe.isnull().sum())


# Aykırı Değerleri Baskılama


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=q1, q3=q3)
    dataframe[variable] = dataframe[variable].astype(float)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in dataframe.columns:
    print(col, check_outlier(dataframe, col))
    if check_outlier(dataframe, col):
        replace_with_thresholds(dataframe, col)

for col in dataframe.columns:
    print(col, check_outlier(dataframe, col))


# Adım 2: Yeni değişkenler oluşturunuz.

# --- Yaş Kategorisi ---
dataframe.loc[(dataframe["Age"] >= 21) & (dataframe["Age"] < 50), "NEW_AGE_CAT"] = "mature"
dataframe.loc[(dataframe["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

# --- BMI Kategorisi ---
# Dünya Sağlık Örgütü sınıflandırması
dataframe.loc[(dataframe["BMI"] < 18.5), "NEW_BMI"] = "Underweight"
dataframe.loc[(dataframe["BMI"] >= 18.5) & (dataframe["BMI"] < 25), "NEW_BMI"] = "Healthy"
dataframe.loc[(dataframe["BMI"] >= 25) & (dataframe["BMI"] < 30), "NEW_BMI"] = "Overweight"
dataframe.loc[(dataframe["BMI"] >= 30), "NEW_BMI"] = "Obese"

# --- Glikoz Kategorisi ---
dataframe.loc[(dataframe["Glucose"] < 70), "NEW_GLUCOSE"] = "Low"
dataframe.loc[(dataframe["Glucose"] >= 70) & (dataframe["Glucose"] < 100), "NEW_GLUCOSE"] = "Normal"
dataframe.loc[(dataframe["Glucose"] >= 100) & (dataframe["Glucose"] < 126), "NEW_GLUCOSE"] = "Prediabetes"
dataframe.loc[(dataframe["Glucose"] >= 126), "NEW_GLUCOSE"] = "Diabetes"

# --- Yaş + BMI Etkileşimi ---
dataframe.loc[(dataframe["BMI"] < 18.5) & (dataframe["Age"] < 50), "NEW_AGE_BMI_NOM"] = "underweightmature"
dataframe.loc[(dataframe["BMI"] < 18.5) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
dataframe.loc[
    (dataframe["BMI"] >= 18.5) & (dataframe["BMI"] < 25) & (dataframe["Age"] < 50), "NEW_AGE_BMI_NOM"] = "healthymature"
dataframe.loc[(dataframe["BMI"] >= 18.5) & (dataframe["BMI"] < 25) & (
            dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
dataframe.loc[(dataframe["BMI"] >= 25) & (dataframe["BMI"] < 30) & (
            dataframe["Age"] < 50), "NEW_AGE_BMI_NOM"] = "overweightmature"
dataframe.loc[(dataframe["BMI"] >= 25) & (dataframe["BMI"] < 30) & (
            dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
dataframe.loc[(dataframe["BMI"] >= 30) & (dataframe["Age"] < 50), "NEW_AGE_BMI_NOM"] = "obesemature"
dataframe.loc[(dataframe["BMI"] >= 30) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

# --- Yaş + Glikoz Etkileşimi ---
dataframe.loc[(dataframe["Glucose"] < 100) & (dataframe["Age"] < 50), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
dataframe.loc[(dataframe["Glucose"] < 100) & (dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
dataframe.loc[(dataframe["Glucose"] >= 100) & (dataframe["Glucose"] < 126) & (
            dataframe["Age"] < 50), "NEW_AGE_GLUCOSE_NOM"] = "prediabetesmature"
dataframe.loc[(dataframe["Glucose"] >= 100) & (dataframe["Glucose"] < 126) & (
            dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "prediabetessenior"
dataframe.loc[(dataframe["Glucose"] >= 126) & (dataframe["Age"] < 50), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
dataframe.loc[(dataframe["Glucose"] >= 126) & (dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"


# --- İnsulin Kategorisi ---
def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"


dataframe["NEW_INSULIN_SCORE"] = dataframe.apply(set_insulin, axis=1)

# --- Glikoz * İnsulin Etkileşimi ---
dataframe["NEW_GLUCOSE_INSULIN"] = dataframe["Glucose"] * dataframe["Insulin"]

# --- Yaş * BMI Sayısal Etkileşimi ---
dataframe["NEW_AGE_BMI"] = dataframe["Age"] * dataframe["BMI"]


# Glikoz / İnsülin oranı (insülin direnci göstergesi)
dataframe["NEW_GLUCOSE_INSULIN_RATIO"] = dataframe["Glucose"] / (dataframe["Insulin"] + 1)

# BMI * Pedigree (genetik + obezite riski)
dataframe["NEW_BMI_PEDIGREE"] = dataframe["BMI"] * dataframe["DiabetesPedigreeFunction"]

# Tansiyon * Yaş (yaşla artan tansiyon riski)
dataframe["NEW_BP_AGE"] = dataframe["BloodPressure"] * dataframe["Age"]

# Hamilelik / Yaş oranı
dataframe["NEW_PREG_AGE"] = dataframe["Pregnancies"] / dataframe["Age"]

# Log dönüşümleri (çarpık dağılımlı sütunlar için)
dataframe["NEW_INSULIN_LOG"] = np.log1p(dataframe["Insulin"])
dataframe["NEW_PEDIGREE_LOG"] = np.log1p(dataframe["DiabetesPedigreeFunction"])
dataframe["NEW_AGE_LOG"] = np.log1p(dataframe["Age"])
dataframe["NEW_PREG_LOG"] = np.log1p(dataframe["Pregnancies"])

# Yüksek riskli kombinasyon flag'leri
dataframe["NEW_HIGH_RISK"] = ((dataframe["Glucose"] >= 126) & (dataframe["BMI"] >= 30)).astype(int)
dataframe["NEW_ELDERLY_OBESE"] = ((dataframe["Age"] >= 50) & (dataframe["BMI"] >= 30)).astype(int)
dataframe["NEW_HIGH_GLUCOSE_INSULIN"] = ((dataframe["Glucose"] >= 126) & (dataframe["Insulin"] > 166)).astype(int)

# Ordinal kategoriler (sayısal kodlanmış)
dataframe.loc[dataframe["Glucose"] < 70, "NEW_GLUCOSE_CAT"] = 0
dataframe.loc[(dataframe["Glucose"] >= 70) & (dataframe["Glucose"] < 100), "NEW_GLUCOSE_CAT"] = 1
dataframe.loc[(dataframe["Glucose"] >= 100) & (dataframe["Glucose"] < 126), "NEW_GLUCOSE_CAT"] = 2
dataframe.loc[dataframe["Glucose"] >= 126, "NEW_GLUCOSE_CAT"] = 3

dataframe.loc[dataframe["BMI"] < 18.5, "NEW_BMI_CAT"] = 0
dataframe.loc[(dataframe["BMI"] >= 18.5) & (dataframe["BMI"] < 25), "NEW_BMI_CAT"] = 1
dataframe.loc[(dataframe["BMI"] >= 25) & (dataframe["BMI"] < 30), "NEW_BMI_CAT"] = 2
dataframe.loc[dataframe["BMI"] >= 30, "NEW_BMI_CAT"] = 3

dataframe.loc[dataframe["Pregnancies"] == 0, "NEW_PREG_CAT"] = 0
dataframe.loc[(dataframe["Pregnancies"] >= 1) & (dataframe["Pregnancies"] <= 3), "NEW_PREG_CAT"] = 1
dataframe.loc[(dataframe["Pregnancies"] >= 4) & (dataframe["Pregnancies"] <= 7), "NEW_PREG_CAT"] = 2
dataframe.loc[dataframe["Pregnancies"] > 7, "NEW_PREG_CAT"] = 3

dataframe.loc[dataframe["DiabetesPedigreeFunction"] < 0.3, "NEW_PEDIGREE_CAT"] = 0
dataframe.loc[(dataframe["DiabetesPedigreeFunction"] >= 0.3) &
              (dataframe["DiabetesPedigreeFunction"] < 0.6), "NEW_PEDIGREE_CAT"] = 1
dataframe.loc[dataframe["DiabetesPedigreeFunction"] >= 0.6, "NEW_PEDIGREE_CAT"] = 2

dataframe.loc[dataframe["BloodPressure"] < 60, "NEW_BP_CAT"] = 0
dataframe.loc[(dataframe["BloodPressure"] >= 60) &
              (dataframe["BloodPressure"] < 80), "NEW_BP_CAT"] = 1
dataframe.loc[dataframe["BloodPressure"] >= 80, "NEW_BP_CAT"] = 2

print("\n>> Oluşturulan yeni değişkenler:")
new_cols = [col for col in dataframe.columns if col.startswith("NEW_")]
for col in new_cols:
    print(f"  {col}")

print(f"\n>> Yeni shape: {dataframe.shape}")

print("\n>> Yeni değişkenler ile Outcome ilişkisi:")
for col in new_cols:
    if dataframe[col].dtype == "O":
        print(f"\n--- {col} ---")
        print(dataframe.groupby(col).agg({"Outcome": ["mean", "count"]}))


# Adım 3: Encoding işlemlerini gerçekleştiriniz.

# Label Encoding (2 sınıflı kategorik değişkenler)
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and dataframe[col].nunique() == 2]

for col in binary_cols:
    dataframe = label_encoder(dataframe, col)

dataframe.head()

# One-Hot Encoding (2'den fazla sınıflı kategorik değişkenler)

cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Outcome"]]


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe


dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=True)

dataframe.head()

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
num_cols = [col for col in num_cols if col not in ["Outcome"]]

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

dataframe.head()


# Adım 5: Model oluşturunuz.
# Random Forest
y = dataframe["Outcome"]
X = dataframe.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")


# Feature importance tablosu
feature_imp = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print(feature_imp.to_string())


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# XGBoost Modeli
xgb_model = XGBClassifier(random_state=46, eval_metric="logloss").fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Karşılaştırma Tablosu
print("=" * 55)
print(f"{'Metrik':<12} {'Random Forest':>18} {'XGBoost':>18}")
print("=" * 55)
print(f"{'Accuracy':<12} {0.90:>18.3f} {round(accuracy_score(y_test, y_pred_xgb), 3):>18.3f}")
print(f"{'Recall':<12} {0.892:>18.3f} {round(recall_score(y_test, y_pred_xgb), 3):>18.3f}")
print(f"{'Precision':<12} {0.81:>18.3f} {round(precision_score(y_test, y_pred_xgb), 3):>18.3f}")
print(f"{'F1':<12} {0.85:>18.3f} {round(f1_score(y_test, y_pred_xgb), 3):>18.3f}")
print(f"{'AUC':<12} {0.90:>18.3f} {round(roc_auc_score(y_test, y_pred_xgb), 3):>18.3f}")
print("=" * 55)

# Feature Importance karşılaştırması
feature_imp_xgb = pd.DataFrame({
    "Feature": X_train.columns,
    "XGBoost_Importance": xgb_model.feature_importances_
}).sort_values("XGBoost_Importance", ascending=False)

print("\nXGBoost Top 10 Feature:")
print(feature_imp_xgb.head(10).to_string(index=False))

#Random Forest & XGBoost Karşılaştırma Grafiği

metrics = ["Accuracy", "Recall", "Precision", "F1", "AUC"]
rf_scores = [0.900, 0.892, 0.810, 0.850, 0.900]
xgb_scores = [0.887, 0.802, 0.867, 0.833, 0.868]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, rf_scores, width, label="Random Forest", color="#2ecc71")
bars2 = ax.bar(x + width/2, xgb_scores, width, label="XGBoost", color="#3498db")

ax.set_ylim(0.75, 0.95)
ax.set_ylabel("Skor")
ax.set_title("Random Forest vs XGBoost — Model Karşılaştırması")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Değerleri bar üzerine yaz
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()




