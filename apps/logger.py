import logging

def activate_logger():
    '''
    Activating Logger for Monitoring.
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', "%d.%m.%Y %H:%M:%S")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    #Only add handler the first time
    if len(logger.handlers) == 0:
        logger.addHandler(handler)
    
    # Disable propagation to prevent logs from bubbling up to the parent logger(s)  
    logger.propagate = False  
    
    return logger