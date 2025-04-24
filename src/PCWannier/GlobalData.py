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
        self.gradient = None

        self.incar_list: list = []
        self.state_collection_list: list = []
        self.m_set_list: list = []
        self.state_initializer_list: list = []
        self.gradient_list: list = []
        

    def push_incar(self, incar=None):
        self.incar_list.append(incar)
        self.incar = incar

    def push_state_collection(self, state_collection=None):
        self.state_collection_list.append(state_collection)
        self.state_collection = state_collection

    def push_m_set(self, m_set=None):
        self.m_set_list.append(m_set)
        self.m_set = m_set

    def push_state_initializer(self, state_initializer=None):
        self.state_initializer_list.append(state_initializer)
        self.state_initializer = state_initializer
    
    def push_gradient(self, gradient=None):
        self.gradient_list.append(gradient)
        self.gradient = gradient

    def select(self, id: int=0):
        num = [len(self.incar_list), len(self.state_collection), len(self.m_set), len(self.state_initializer), len(self.gradient)]
        max_num = max(num)
        for _ in range(max_num - num[0]):
            self.push_incar()
        for _ in range(max_num - num[1]):
            self.push_state_collection()
        for _ in range(max_num - num[2]):
            self.push_m_set()
        for _ in range(max_num - num[3]):
            self.push_state_initializer()
        for _ in range(max_num - num[4]):
            self.push_gradient()

        self.incar = self.incar_list[id]
        self.state_collection = self.state_collection_list[id]
        self.m_set = self.m_set_list[id]
        self.state_initializer = self.state_initializer_list[id]
        self.gradient = self.gradient_list[id]


global_data = GlobalData()