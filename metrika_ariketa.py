import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("etxeak.csv")
X = df[["prezioa", "azalera_m2"]]
y = df["gela_kopurua"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sc = StandardScaler()
X_train_esk = sc.fit_transform(X_train)
X_test_esk = sc.transform(X_test)

emaitzak = []
for k in range(1, 7):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_esk, y_train)
    pred_tr = knn.predict(X_train_esk)
    pred_te = knn.predict(X_test_esk)
    emaitzak.append({
        "k": k,
        "R2_entren": round(r2_score(y_train, pred_tr), 3),
        "R2_proba": round(r2_score(y_test, pred_te), 3),
        "MAE_entren": round(mean_absolute_error(y_train, pred_tr), 3),
        "MAE_proba": round(mean_absolute_error(y_test, pred_te), 3),
    })

taula = pd.DataFrame(emaitzak)
print(taula)