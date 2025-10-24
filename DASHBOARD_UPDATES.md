# 📊 ACTUALIZACIONES DEL DASHBOARD - METODOLOGÍA

## ✅ CAMBIOS REALIZADOS

Se actualizó completamente la sección **"ℹ️ Metodología de Clasificación (Detalles Técnicos)"** del dashboard para reflejar el nuevo método de clasificación basado en Embeddings.

---

## 🔄 CAMBIOS EN LA METODOLOGÍA

### ANTES (Zero-Shot):
```
Método: Zero-Shot Classification con facebook/bart-large-mnli
Umbrales: Score ML ≥ 50 (principal), ≥ 40 (secundario)
Niveles: Alta (≥65), Media (50-64), Tentativa (<50)
Tiempo: ~43 minutos
```

### DESPUÉS (Embeddings):
```
Método: Embeddings + Similitud Coseno
Modelo: paraphrase-multilingual-MiniLM-L12-v2
Umbrales: Similitud ≥ 0.35 (principal), ≥ 0.30 (secundario)
Niveles: Alta (≥50%), Media (40-50%), Baja (35-40%), Tentativa (<35%)
Tiempo: ~30 segundos
```

---

## 📝 DETALLES DE LOS CAMBIOS

### 1. Método de Clasificación (Columna Izquierda)

**✅ ACTUALIZADO:**
- Método: Embeddings + Similitud Coseno
- Modelo: `paraphrase-multilingual-MiniLM-L12-v2`
- Clasificación basada en similitud semántica real
- Multilingüe (español + inglés)

**Umbrales de Similitud:**
- Línea Principal: Similitud ≥ 0.35 (35%)
- Línea Secundaria: Similitud ≥ 0.30 (30%)
- Multi-línea: Artículos con múltiples líneas ≥ umbral secundario

**Datos Utilizados:**
- Título completo del artículo
- Abstract (92.5% disponibles)
- MeSH terms (vocabulario controlado)
- Keywords del autor

### 2. Niveles de Confianza (Columna Derecha)

**✅ ACTUALIZADO:**
- 🟢 **Alta** (≥50%): Mostrado dinámicamente
- 🟡 **Media** (40-50%): Mostrado dinámicamente
- 🟠 **Baja** (35-40%): Nivel nuevo añadido
- 🔴 **Tentativa** (<35%): Mostrado dinámicamente

**Características del Modelo:**
- Tamaño: ~420 MB
- Arquitectura: Sentence Transformers
- Embeddings: 384 dimensiones
- Velocidad: ~30 segundos (226 artículos)
- Distribución equilibrada sin sesgos

### 3. Nombres de Líneas en Tabs

**✅ ACTUALIZADO:**
- 🧬 Línea 1: Bases Moleculares y Genómica Nutricional
- 🏥 Línea 2: Epidemiología Clínica y Factores de Riesgo
- 👥 Línea 3: Salud Poblacional y Políticas Públicas

(Anteriormente eran nombres genéricos como "Genómica Nutricional", "Salud Pública", etc.)

### 4. Mensaje de Warning

**✅ ACTUALIZADO:**
Cuando no hay datos de clasificación, ahora muestra:
```
Ejecuta primero el script de clasificación:
1. python src/embeddings_classifier.py - Clasificación con Embeddings
2. python src/convert_embeddings_to_dashboard.py - Conversión a formato dashboard
```

(Anteriormente refería a ml_zero_shot_classifier.py y aggregate_ml_classifications.py)

---

## 🎯 IMPACTO EN EL USUARIO

### Lo que verá el usuario:

1. **Metodología actualizada:** Información precisa sobre el método de embeddings
2. **Umbrales claros:** Porcentajes de similitud fáciles de entender
3. **4 niveles de confianza:** Más granularidad (alta, media, baja, tentativa)
4. **Datos utilizados:** Transparencia sobre qué información se usa
5. **Características técnicas:** Modelo más pequeño (~420 MB vs ~380 MB) pero más rápido
6. **Nombres correctos:** Títulos completos de las líneas de investigación

### Beneficios:

✅ **Transparencia:** Los usuarios entienden cómo funciona la clasificación  
✅ **Precisión:** Información técnica actualizada y correcta  
✅ **Claridad:** Umbrales expresados como porcentajes de similitud  
✅ **Completitud:** Muestra qué datos se utilizan (título, abstract, MeSH, keywords)  
✅ **Confianza:** Información sobre el rendimiento del modelo

---

## 📊 VISUALIZACIÓN EN EL DASHBOARD

La sección expandible "ℹ️ Metodología de Clasificación (Detalles Técnicos)" ahora muestra:

```
┌─────────────────────────────────────────────────────────────────┐
│ Método de Clasificación:          │ Niveles de Confianza:        │
│ • Embeddings + Similitud Coseno   │ • 🟢 Alta (≥50%): XX arts.   │
│ • Modelo: paraphrase-multi...     │ • 🟡 Media (40-50%): XX arts.│
│ • Multilingüe                     │ • 🟠 Baja (35-40%): XX arts. │
│                                   │ • 🔴 Tentativa (<35%): XX    │
│ Umbrales de Similitud:            │                              │
│ • Principal: ≥ 0.35 (35%)         │ Características del Modelo:  │
│ • Secundaria: ≥ 0.30 (30%)        │ • Tamaño: ~420 MB            │
│ • Multi-línea: ≥ secundario       │ • Arquitectura: SentenceT... │
│                                   │ • Embeddings: 384 dims       │
│ Datos Utilizados:                 │ • Velocidad: ~30 segundos    │
│ • Título completo                 │ • Distribución equilibrada   │
│ • Abstract (92.5%)                │                              │
│ • MeSH terms                      │                              │
│ • Keywords                        │                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 DASHBOARD ACTUALIZADO

**Estado:** ✅ Corriendo en http://localhost:8501  
**Archivos modificados:** `src/app.py`  
**Líneas actualizadas:** 2481-2514, 2541-2547, 2552-2556  

---

## ✅ PRÓXIMOS PASOS

El dashboard está completamente actualizado y listo para uso:

1. ✅ Metodología actualizada a Embeddings
2. ✅ Nombres correctos de líneas
3. ✅ Umbrales y confianza actualizados
4. ✅ Datos utilizados documentados
5. ✅ Scripts correctos en mensajes de error

**No se requieren cambios adicionales en esta sección.**

---

Fecha: 2025-10-24  
Dashboard: http://localhost:8501  
Método: Embeddings + Similitud Coseno  
