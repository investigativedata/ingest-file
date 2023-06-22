from ftmstore import settings as fts
from servicelayer import env
from servicelayer import settings as sls

TESTING = False

CONVERT_TIMEOUT = env.to_int("INGESTORS_CONVERT_TIMEOUT", 7200)  # 2 hrs
CONVERT_RETRIES = env.to_int("INGESTORS_CONVERT_RETRIES", 256)

# Enable (expensive!) Google Cloud API
OCR_VISION_API = env.to_bool("INGESTORS_OCR_VISION_API", False)

# Geonames data file
GEONAMES_PATH = env.get("INGESTORS_GEONAMES_PATH", "/ingestors/data/geonames.txt")

# FastText lid model file
LID_MODEL_PATH = env.get("INGESTORS_LID_MODEL_PATH", "/ingestors/data/lid.176.ftz")

# Disable entity extraction
ANALYZE_ENTITIES = env.to_bool("INGESTORS_ANALYZE_ENTITIES", True)

# List available NER models
NER_MODELS = {
    "eng": "en_core_web_lg",
    "deu": "de_core_news_lg",
    "fra": "fr_core_news_lg",
    "spa": "es_core_news_lg",
    "dan": "da_core_news_lg",
    "ukr": "uk_core_news_lg",
    "rus": "ru_core_news_lg",
    "por": "pt_core_news_lg",
    "ron": "ro_core_news_lg",
    "mkd": "mk_core_news_lg",
    "ell": "el_core_news_lg",
    "pol": "pl_core_news_lg",
    "ita": "it_core_news_lg",
    "lit": "lt_core_news_lg",
    "nld": "nl_core_news_lg",
    "nob": "nb_core_news_lg",
    "nor": "nb_core_news_lg",
    "swe": "sv_core_news_lg",
}

# FastText type prediction model file
NER_TYPE_MODEL_PATH = env.get(
    "INGESTORS_TYPE_MODEL_PATH", "/models/model_type_prediction.ftz"
)

# Use the environment variable set in aleph.env
fts.DATABASE_URI = env.get(
    "FTM_STORE_URI", env.get("ALEPH_DATABASE_URI", fts.DATABASE_URI)
)

# Also store cached values in the SQL database
sls.TAGS_DATABASE_URI = fts.DATABASE_URI
