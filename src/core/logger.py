import logging
import logging.config
import sys

def setup_logging(level=logging.INFO):
    """
    Configure global logging with noise suppression for third-party libraries.
    """
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': level,
                'formatter': 'standard',
                'filename': 'noa.log',
                'mode': 'a',
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'file'],
                'level': level,
                'propagate': True
            },
            # Silence noisy libraries
            'httpx': {'level': 'WARNING'},
            'httpcore': {'level': 'WARNING'},
            'urllib3': {'level': 'WARNING'},
            'uvicorn': {'level': 'INFO'},
            'uvicorn.access': {'level': 'WARNING'}, # Reduce access logs
            'multipart': {'level': 'WARNING'},
            'watchfiles': {'level': 'WARNING'},
            'llama_cpp': {'level': 'ERROR'}, # Suppress LLM low-level logs
        }
    }

    logging.config.dictConfig(logging_config)
    
    # Force set specific loggers just in case
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logging.getLogger("noa_core")
