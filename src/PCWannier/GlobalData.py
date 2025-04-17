class GlobalData:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance
    
    def __init__(self) -> None:
        self.threads = 1
        self.incar = None
        self.state_collection = None

        self.m_set = None
        self.state_initializer = None
    
    def init(self):
        self.threads = 1
        self.incar = None
        self.state_collection = None

        self.m_set = None

global_data = GlobalData()