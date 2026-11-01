import re
from pathlib import Path

import pandas as pd
import numpy as np

RAW_PATH = Path("/home/val/Documentos/vale221/practica1.py/__pycache__/base-datos/denue_sucia.csv")
OUT_DIR = RAW_PATH.parent / "denue_normalizado"
OUT_DIR.mkdir(exist_ok=True)


def clean_df_columns(df):
    cols = df.columns.astype(str).str.strip()
    cols = [re.sub(r"\s+", "_", c).lower() for c in cols]
    cols = [c.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n") for c in cols]
    cols = [c.replace("(", "").replace(")", "").replace("/", "_").replace("-", "_") for c in cols]
    cols = [re.sub(r"__+", "_", c) for c in cols]
    return cols


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def split_contacts(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", ""}:
        return []
    parts = re.split(r"[;,]", text)
    return [p.strip() for p in parts if p.strip()]


def es_empresa(x):
    x = str(x).strip()
    if x == "":
        return True
    xup = x.upper()
    if any(k in xup for k in ["S.A.", "SA", "S DE RL", "LTDA", "SOCIEDAD", "INDUSTRIAL", "COMERCIAL", "SERVICIOS", "AC"]):
        return True
    if len(x.split()) >= 3:
        return True
    return False


def separar_nombre_apellido(x):
    x = str(x).strip()
    if x == "":
        return "", ""
    if es_empresa(x):
        return x, ""
    partes = x.split()
    if len(partes) >= 2:
        return partes[0], " ".join(partes[1:])
    return x, ""


df = pd.read_csv(RAW_PATH, encoding="latin1", dtype=str)
df.columns = clean_df_columns(df)

rename_map = {
    "id": "id_original",
    "clee": "clee",
    "nombre_de_la_unidad_economica": "nombre_unidad",
    "razon_social": "razon_social",
    "codigo_de_la_clase_de_actividad_scian": "codigo_scian",
    "nombre_de_clase_de_la_actividad": "nombre_clase",
    "descripcion_estrato_personal_ocupado": "descripcion_estrato",
    "tipo_de_vialidad": "tipo_vialidad",
    "nombre_de_la_vialidad": "nombre_vialidad",
    "tipo_de_entre_vialidad_1": "tipo_entre_vialidad_1",
    "nombre_de_entre_vialidad_1": "nombre_entre_vialidad_1",
    "tipo_de_entre_vialidad_2": "tipo_entre_vialidad_2",
    "nombre_de_entre_vialidad_2": "nombre_entre_vialidad_2",
    "tipo_de_entre_vialidad_3": "tipo_entre_vialidad_3",
    "nombre_de_entre_vialidad_3": "nombre_entre_vialidad_3",
    "numero_exterior_o_kilometro": "numero_exterior",
    "letra_exterior": "letra_exterior",
    "tipo_de_asentamiento_humano": "tipo_asentamiento",
    "nombre_de_asentamiento_humano": "nombre_asentamiento",
    "codigo_postal": "codigo_postal",
    "clave_entidad": "clave_entidad",
    "entidad_federativa": "entidad_federativa",
    "clave_municipio": "clave_municipio",
    "municipio": "municipio",
    "clave_localidad": "clave_localidad",
    "localidad": "localidad",
    "numero_de_telefono": "numero_telefono",
    "correo_electronico": "correo_electronico",
    "sitio_en_internet": "sitio_web",
    "tipo_de_establecimiento": "tipo_establecimiento",
    "latitud": "latitud",
    "longitud": "longitud",
}

df = df.rename(columns=rename_map)

for c in df.columns:
    df[c] = df[c].apply(normalize_text)

# 1FN: columnas atómicas y propietario en nombre/apellido
df["propietario_raw"] = np.where(df["razon_social"].str.strip() == "", df["nombre_unidad"], df["razon_social"])
propietario = df[["propietario_raw"]].drop_duplicates().reset_index(drop=True)
propietario[["nombre_propietario", "apellido_propietario"]] = propietario["propietario_raw"].apply(
    lambda x: pd.Series(separar_nombre_apellido(x))
)
propietario.insert(0, "id_propietario", range(1, len(propietario) + 1))

# 2FN: catálogos sin duplicados
cat_entidad = df[["clave_entidad", "entidad_federativa"]].drop_duplicates().reset_index(drop=True)
cat_municipio = df[["clave_municipio", "municipio", "clave_entidad"]].drop_duplicates().reset_index(drop=True)
cat_localidad = df[["clave_localidad", "localidad", "clave_municipio"]].drop_duplicates().reset_index(drop=True)
cat_scian = df[["codigo_scian", "nombre_clase"]].drop_duplicates().reset_index(drop=True)
cat_estrato = df[["descripcion_estrato"]].drop_duplicates().dropna().reset_index(drop=True)
cat_estrato.insert(0, "id_estrato", range(1, len(cat_estrato) + 1))
cat_asentamiento = df[["tipo_asentamiento", "nombre_asentamiento", "codigo_postal"]].drop_duplicates().dropna().reset_index(drop=True)
cat_asentamiento.insert(0, "id_asentamiento", range(1, len(cat_asentamiento) + 1))

# 3FN: entidad principal + relación muchos a muchos
unidad_economica = df[[
    "id_original",
    "nombre_unidad",
    "razon_social",
    "codigo_scian",
    "descripcion_estrato",
    "tipo_vialidad",
    "nombre_vialidad",
    "numero_exterior",
    "letra_exterior",
    "tipo_asentamiento",
    "nombre_asentamiento",
    "codigo_postal",
    "clave_entidad",
    "clave_municipio",
    "clave_localidad",
    "latitud",
    "longitud",
    "propietario_raw"
]].copy()
unidad_economica = unidad_economica.rename(columns={"id_original": "id_unidad"})
unidad_economica = unidad_economica.merge(
    propietario[["propietario_raw", "id_propietario"]],
    on="propietario_raw",
    how="left"
)
unidad_economica = unidad_economica.drop(columns=["propietario_raw"])
unidad_economica = unidad_economica.drop_duplicates().reset_index(drop=True)

rows_contacto = []
for _, r in df.iterrows():
    id_unidad = r.get("id_original", "")
    for telefono in split_contacts(r.get("numero_telefono", "")):
        rows_contacto.append({"id_unidad": id_unidad, "tipo_contacto": "telefono", "valor_contacto": telefono})
    for correo in split_contacts(r.get("correo_electronico", "")):
        rows_contacto.append({"id_unidad": id_unidad, "tipo_contacto": "correo", "valor_contacto": correo})
    for web in split_contacts(r.get("sitio_web", "")):
        rows_contacto.append({"id_unidad": id_unidad, "tipo_contacto": "sitio_web", "valor_contacto": web})

detalle_contacto = pd.DataFrame(rows_contacto).drop_duplicates().reset_index(drop=True)
if not detalle_contacto.empty:
    detalle_contacto.insert(0, "id_detalle_contacto", range(1, len(detalle_contacto) + 1))

rows_vial = []
for _, r in df.iterrows():
    id_unidad = r.get("id_original", "")
    for tipo, nombre in [
        (r.get("tipo_entre_vialidad_1", ""), r.get("nombre_entre_vialidad_1", "")),
        (r.get("tipo_entre_vialidad_2", ""), r.get("nombre_entre_vialidad_2", "")),
        (r.get("tipo_entre_vialidad_3", ""), r.get("nombre_entre_vialidad_3", "")),
    ]:
        if str(tipo).strip() or str(nombre).strip():
            rows_vial.append({
                "id_unidad": id_unidad,
                "tipo_entre_vialidad": str(tipo).strip(),
                "nombre_entre_vialidad": str(nombre).strip(),
            })

detalle_unidad_vialidad = pd.DataFrame(rows_vial).drop_duplicates().reset_index(drop=True)
if not detalle_unidad_vialidad.empty:
    detalle_unidad_vialidad.insert(0, "id_detalle_vialidad", range(1, len(detalle_unidad_vialidad) + 1))

# Guardar CSVs
for table_name, table_df in {
    "cat_entidad.csv": cat_entidad,
    "cat_municipio.csv": cat_municipio,
    "cat_localidad.csv": cat_localidad,
    "cat_scian.csv": cat_scian,
    "cat_estrato.csv": cat_estrato,
    "cat_asentamiento.csv": cat_asentamiento,
    "cat_propietario.csv": propietario,
    "unidad_economica.csv": unidad_economica,
    "detalle_contacto.csv": detalle_contacto,
    "detalle_unidad_vialidad.csv": detalle_unidad_vialidad,
}.items():
    table_df.to_csv(OUT_DIR / table_name, index=False)

# Salida en terminal: 3 tablas con título
print("\n====================")
print("1FN")
print("====================")
print(propietario.head(5).to_string(index=False))
print("\n")
print(cat_entidad.head(5).to_string(index=False))

print("\n====================")
print("2FN")
print("====================")
print(cat_municipio.head(5).to_string(index=False))
print("\n")
print(cat_localidad.head(5).to_string(index=False))

print("\n====================")
print("3FN")
print("====================")
print(unidad_economica.head(5).to_string(index=False))
print("\n")
print(detalle_contacto.head(5).to_string(index=False))
print("\n")
print(detalle_unidad_vialidad.head(5).to_string(index=False))

print("\n✅ Script ejecutado correctamente")
print(f"Archivos generados en: {OUT_DIR}")
