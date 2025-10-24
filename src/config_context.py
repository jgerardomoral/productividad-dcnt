"""
Configuración y datos contextuales para el Dashboard DCNT-UdeG
Datos extraídos del documento de pertinencia estratégica
"""

# ============================================================================
# DATOS EPIDEMIOLÓGICOS NACIONALES (ENSANUT 2022-2023)
# ============================================================================

EPIDEMIOLOGIA_MEXICO = {
    "sobrepeso_obesidad_adultos": {
        "valor": 75.2,
        "descripcion": "% adultos con sobrepeso u obesidad",
        "fuente": "ENSANUT 2022",
        "desglose": "38.3% sobrepeso + 36.9% obesidad"
    },
    "diabetes_adultos": {
        "valor": 18.3,
        "descripcion": "% adultos con diabetes",
        "fuente": "ENSANUT 2022",
        "personas": "14.6 millones de personas"
    },
    "obesidad_abdominal": {
        "valor": 81.0,
        "descripcion": "% adultos con obesidad abdominal",
        "fuente": "ENSANUT 2022"
    },
    "desnutricion_infantil": {
        "valor": 12.8,
        "descripcion": "% niños menores de 5 años con baja talla",
        "fuente": "ENSANUT 2022",
        "tendencia": "Estancado desde 2012"
    },
    "anemia_mujeres": {
        "valor": 15.8,
        "descripcion": "% mujeres 15-49 años con anemia",
        "fuente": "ENSANUT 2022",
        "tendencia": "Incrementó de 11.6% (2012) a 15.8% (2022)"
    },
    "mortalidad_diabetes": {
        "valor": 110174,
        "descripcion": "Muertes por diabetes en 2023",
        "fuente": "Sistema de Vigilancia 2024"
    },
    "ultraprocesados": {
        "valor": 46.6,
        "descripcion": "% consumo de alimentos ultraprocesados",
        "fuente": "ENSANUT 2022",
        "tendencia": "Aumentó de 39.5% (2000) a 46.6% (2020)"
    },
    "inseguridad_alimentaria": {
        "valor": 18.2,
        "descripcion": "% población con carencia por acceso a alimentación",
        "fuente": "CONEVAL 2022",
        "personas": "23.4 millones de mexicanos"
    }
}

# ============================================================================
# DATOS EPIDEMIOLÓGICOS JALISCO
# ============================================================================

EPIDEMIOLOGIA_JALISCO = {
    "desnutricion_casos": {
        "2021": 3341,
        "2023": 6284,
        "incremento_porcentual": 88,
        "ranking": "2do lugar nacional",
        "fuente": "Sistema Nacional de Vigilancia Epidemiológica"
    },
    "muertes_obesidad": {
        "total_anual": 42000,
        "asociadas_obesidad": 15000,
        "porcentaje": 30,
        "descripcion": "30% de muertes anuales asociadas a obesidad/sobrepeso"
    },
    "diabetes_casos": {
        "valor": 32770,
        "ranking": "Mayor número de casos hospitalizados a nivel nacional (junto con CDMX)"
    },
    "sin_atencion_nutricional": {
        "valor": 56.5,
        "descripcion": "% diabéticos hospitalizados que NO reciben atención nutricional"
    },
    "actividad_fisica_inadecuada": {
        "valor": 84.3,
        "descripcion": "% diabéticos sin actividad física diaria adecuada (solo 15.7% la hace)"
    },
    "carencia_alimentaria": {
        "personas": 1176459,
        "descripcion": "Personas con carencia de acceso a alimentación",
        "fuente": "CONEVAL 2022"
    }
}

# ============================================================================
# OBJETIVOS DE DESARROLLO SOSTENIBLE (ODS)
# ============================================================================

ODS_CONTEXTO = {
    "ODS_2": {
        "nombre": "Hambre Cero",
        "numero": 2,
        "meta_2_2": "Poner fin a todas las formas de malnutrición",
        "indicadores_mexico": {
            "baja_talla_infantil": "12.8% (estancado desde 2012)",
            "baja_talla_indigena": "27.4% vs 13.9% promedio nacional",
            "sobrepeso_infantil": "7.7% en menores de 5 años",
            "anemia_mujeres": "15.8% (incremento de 11.6% en 2012)",
            "meta_2025": "México proyectado a cumplir solo 1 de 6 metas nutricionales globales"
        },
        "contribucion_dcnt": [
            "Investigación en primeros 1000 días de vida (programación metabólica)",
            "Epigenética y genómica nutricional en población mexicana",
            "Intervenciones basadas en biomarcadores moleculares",
            "Evaluación de programas de suplementación"
        ]
    },
    "ODS_3": {
        "nombre": "Salud y Bienestar",
        "numero": 3,
        "meta_3_4": "Reducir en 1/3 mortalidad prematura por ENT para 2030",
        "situacion_mexico": {
            "diabetes": "18.3% adultos (14.6 millones)",
            "obesidad": "36.9% adultos",
            "sobrepeso_obesidad": "75.2% adultos",
            "hipertension": "47% cumple criterios",
            "proyeccion_2050": "88% con sobrepeso/obesidad sin intervenciones efectivas",
            "mortalidad": "Enfermedades cardíacas, diabetes y tumores son principales causas de muerte"
        },
        "contribucion_dcnt": [
            "Genómica nutricional para identificar poblaciones alto riesgo",
            "Intervenciones personalizadas basadas en perfil genético",
            "Diseño de intervenciones poblacionales escalables",
            "Investigación traslacional clínica para mejorar atención"
        ]
    },
    "ODS_10": {
        "nombre": "Reducción de las Desigualdades",
        "numero": 10,
        "brechas_nutricionales": {
            "baja_talla_indigena": "27.4% vs 13.9% promedio",
            "quintil_bajo_bienestar": "20.8% baja talla vs mucho menor en quintiles altos",
            "region_pacifico_sur": "20% baja talla vs 12.8% promedio",
            "anemia_poblacion_vulnerable": "34.3% en mujeres con menores capacidades económicas"
        },
        "contribucion_dcnt": [
            "Intervenciones culturalmente pertinentes (ej. comunidades Wixárikas)",
            "Investigación con diversidad epistémica",
            "Reducción de inequidades en salud nutricional",
            "Desarrollo de soluciones de bajo costo para poblaciones vulnerables"
        ]
    },
    "ODS_12": {
        "nombre": "Producción y Consumo Responsables",
        "numero": 12,
        "meta_12_3": "Reducir desperdicio alimentario",
        "meta_12_8": "Promover estilos de vida sostenibles",
        "situacion_mexico": {
            "desperdicio_alimentos": "20.4 millones de toneladas anuales (34% producción)",
            "inseguridad_alimentaria": "20.7% población (moderada o severa)",
            "ultraprocesados": "46.6% del consumo total (incremento de 7.1 puntos en 20 años)",
            "bebidas_azucaradas": "27% de muertes por diabetes relacionadas con bebidas azucaradas"
        },
        "contribucion_dcnt": [
            "Evaluación de alimentos tradicionales vs ultraprocesados",
            "Desarrollo de alimentos funcionales con biodiversidad mexicana",
            "Sistemas alimentarios más eficientes y sostenibles",
            "Educación nutricional para cambiar patrones de consumo"
        ]
    }
}

# ============================================================================
# PRONACES (Programas Nacionales Estratégicos)
# ============================================================================

PRONACES_CONTEXTO = {
    "inversion_total": {
        "monto": "1,700 millones de pesos",
        "proyectos": 666,
        "instituciones": 170,
        "personas": "10,000+",
        "periodo": "2023-2025"
    },
    "PRONACE_SALUD": {
        "nombre": "PRONACE Salud",
        "areas_prioritarias": [
            "Alimentación y Salud Integral Comunitaria",
            "Enfermedades Crónicas no Transmisibles",
            "Medicina de Sistemas y Determinantes Moleculares",
            "Ciencia de Datos Aplicada a Salud"
        ],
        "financiamiento": "Hasta 5 millones de pesos anuales durante 5 años",
        "temas_convocatoria_2022": [
            "Alimentación escolar y comunitaria",
            "Patrones alimentarios saludables",
            "Prevención de obesidad",
            "Prevención de enfermedades crónico-degenerativas"
        ],
        "alineacion_dcnt": {
            "Línea 1 (Genómica Nutricional)": "Medicina de sistemas, determinantes moleculares, biomarcadores",
            "Línea 2 (Salud Pública)": "Alimentación comunitaria, investigación desde enfoques comunitarios, intervenciones escalables",
            "Línea 3 (Alimentación y Nutrición)": "Patrones alimentarios, servicios de alimentación, atención clínica"
        }
    },
    "PRONACE_SOBERANIA_ALIMENTARIA": {
        "nombre": "PRONACE Soberanía Alimentaria",
        "pronaii_activos": 38,
        "localidades": 314,
        "organizaciones_comunitarias": 201,
        "modelo": "Transdisciplinario 5 años integrando academia-gobierno-comunidad",
        "demandas_prioritarias": [
            "Estrategias para alimentación segura, saludable, nutritiva y culturalmente adecuada",
            "Educación para la alimentación saludable",
            "Alimentos funcionales",
            "Calidad nutrimental del maíz-tortilla",
            "Circuitos regionales de alimentos"
        ],
        "alineacion_dcnt": {
            "Línea 2 (Salud Pública)": "Educación alimentaria, evaluación de programas comunitarios",
            "Línea 3 (Alimentación y Nutrición)": "Calidad nutrimental, alimentos funcionales, sistemas regionales, ciencias de alimentos"
        }
    }
}

# ============================================================================
# LÍNEAS DE INVESTIGACIÓN DEL DCNT-UdeG
# ============================================================================

LINEAS_INVESTIGACION = {
    "linea_1": {
        "nombre": "Bases Moleculares y Genómica Nutricional de la Nutrición Humana",
        "descripcion": "Interacción entre nutrimentos de la dieta y genes, bases moleculares y genéticas en estados fisiológicos y patológicos",
        "areas_investigacion": [
            "Interacciones gen-dieta",
            "Epigenética nutricional",
            "Biomarcadores metabólicos",
            "Medicina de precisión nutricional",
            "Programación metabólica (primeros 1000 días)",
            "Variabilidad genética en población mexicana"
        ],
        "aplicaciones": [
            "Identificación de poblaciones de alto riesgo",
            "Intervenciones personalizadas basadas en genotipo",
            "Estratificación de riesgo para diabetes, obesidad y enfermedades cardiovasculares",
            "Desarrollo de alimentos funcionales personalizados"
        ],
        "campo_laboral": [
            "Centros de investigación biomédica",
            "Laboratorios de medicina genómica",
            "Industria farmacéutica",
            "Industria de alimentos funcionales",
            "Instituciones de salud pública"
        ]
    },
    "linea_2": {
        "nombre": "Alimentación y Nutrición Humana en Salud Pública",
        "descripcion": "Investigación en salud pública, evaluación de factores personales y ambientales relacionados con el proceso alimentario-nutricio",
        "areas_investigacion": [
            "Determinantes sociales de la salud nutricional",
            "Intervenciones poblacionales escalables",
            "Evaluación de programas y políticas públicas",
            "Ambientes alimentarios",
            "Educación nutricional masiva",
            "Sistemas de vigilancia epidemiológica"
        ],
        "aplicaciones": [
            "Diseño de estudios de implementación",
            "Evaluación de impacto de etiquetado frontal",
            "Intervenciones para cambiar ambientes alimentarios",
            "Reducción de inequidades en salud nutricional",
            "Campañas de salud pública basadas en evidencia"
        ],
        "campo_laboral": [
            "Secretarías de Salud (federal y estatal)",
            "IMSS / ISSSTE",
            "Organizaciones internacionales (OPS, UNICEF, FAO)",
            "ONGs en nutrición y salud pública",
            "Consultoría en políticas públicas"
        ]
    },
    "linea_3": {
        "nombre": "Alimentación y Nutrición Humana",
        "descripcion": "Proceso alimentario-nutricio integral, generación de conocimientos del campo clínico, ciencias de alimentos, servicios de alimentación y educación en nutrición",
        "areas_investigacion": [
            "Ciencias y tecnología de alimentos",
            "Nutrición clínica en estados patológicos",
            "Servicios de alimentación institucional",
            "Alimentos funcionales",
            "Inocuidad y calidad nutrimental",
            "Biodiversidad alimentaria mexicana"
        ],
        "aplicaciones": [
            "Formulación de alimentos para poblaciones específicas",
            "Protocolos de atención nutricional clínica",
            "Desarrollo de alimentos funcionales (nopal, amaranto, chía)",
            "Evaluación de calidad nutrimental maíz-tortilla",
            "Diseño de servicios de alimentación hospitalaria/escolar"
        ],
        "campo_laboral": [
            "Industria alimentaria",
            "Hospitales de tercer nivel",
            "Servicios de alimentación institucional",
            "Desarrollo de productos alimenticios",
            "Consultoría en nutrición clínica"
        ]
    }
}

# ============================================================================
# PERTINENCIA REGIONAL
# ============================================================================

REGION_OCCIDENTE = {
    "poblacion": "15,063,844 habitantes",
    "pib_nacional": "11%",
    "estados": ["Jalisco", "Colima", "Michoacán", "Nayarit"],
    "indice_progreso_social": 62.8,  # Por debajo del promedio nacional
    "lider_regional": {
        "estado": "Jalisco",
        "ips": 69.02,
        "economia": "2da economía nacional",
        "pib_regional": "65.7% del PIB de región Occidente"
    },
    "deficit_formacion": "Único doctorado en nutrición traslacional en región Occidente (PNPC-CONAHCYT)"
}

INFRAESTRUCTURA_UDG = {
    "cucs": {
        "investigadores_sni": 131,
        "profesores_tiempo_completo": 489,
        "articulos_anuales": 270,
        "proyectos_activos": 500
    },
    "inhu": {
        "años_operacion": "30+ años (desde 1995)",
        "maestria_pnpc": "Consolidada desde 2009",
        "generaciones_formadas": 12,
        "experiencia": "Investigación materno-infantil, primeros 1000 días"
    },
    "hospital_civil": {
        "investigadores_sni": 73,
        "publicaciones_anuales": 270,
        "proyectos_investigacion": 500,
        "recursos_humanos_formados": "4,500 anuales en 73 programas"
    },
    "cmno_imss": {
        "usuarios_potenciales": "17+ millones (30% derechohabientes IMSS nacional)",
        "infraestructura": "Centro de Investigación Biomédica de Occidente (CIBO)"
    },
    "ss_jalisco": {
        "centros_salud": 580,
        "hospitales": 41,
        "capacidades": "Telemedicina, big data (50 años información), sistema urgencias mejor de Latinoamérica"
    }
}

# ============================================================================
# INVESTIGACIÓN TRASLACIONAL - CONTINUUM T0-T4
# ============================================================================

INVESTIGACION_TRASLACIONAL = {
    "T0": {
        "fase": "Investigación Básica",
        "descripcion": "Mecanismos moleculares, interacciones gen-dieta, biomarcadores",
        "linea_dcnt": "Línea 1 (Genómica Nutricional)"
    },
    "T1": {
        "fase": "Traslación a Humanos",
        "descripcion": "Estudios clínicos controlados, ensayos de intervenciones dietéticas guiadas por genotipo",
        "linea_dcnt": "Líneas 1 y 3"
    },
    "T2": {
        "fase": "Traslación a Pacientes",
        "descripcion": "Desarrollo de guías clínicas basadas en evidencia, protocolos de atención nutricional",
        "linea_dcnt": "Línea 3 (Alimentación y Nutrición)"
    },
    "T3": {
        "fase": "Traslación a Práctica",
        "descripcion": "Implementación en sistemas de salud reales, evaluación de efectividad",
        "linea_dcnt": "Línea 2 (Salud Pública)"
    },
    "T4": {
        "fase": "Traslación a Población",
        "descripcion": "Escalamiento de intervenciones exitosas, políticas públicas basadas en evidencia",
        "linea_dcnt": "Línea 2 (Salud Pública)"
    }
}

# ============================================================================
# CASOS ESPECÍFICOS DE POBLACIONES VULNERABLES
# ============================================================================

POBLACIONES_VULNERABLES = {
    "comunidades_wixarikas": {
        "ubicacion": "Norte de Jalisco (Mezquitic)",
        "caracteristicas": "Alto aislamiento geográfico en Sierra Madre Occidental",
        "problemas": "Problemas alimentario-nutricionales infantiles documentados desde 1999",
        "colaboracion_udg": "25+ años, invitación del Consejo de Salud Comunitaria Werika",
        "necesidad": "Investigadores con competencias en nutrición traslacional e interculturalidad"
    },
    "agua_caliente_poncitlan": {
        "ubicacion": "Agua Caliente, Poncitlán, Jalisco",
        "problema": "Alta prevalencia de desnutrición + enfermedad renal crónica de etiología desconocida",
        "situacion": "Calificada como de gravedad y urgencia",
        "necesidad": "Investigación traslacional urgente"
    }
}
