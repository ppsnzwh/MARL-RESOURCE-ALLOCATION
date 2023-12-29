class Req:
    def __init__(self, dict):
        self.id = int(dict.get("id"))
        self.cpu = int(dict.get("cpu"))
        self.memory = int(dict.get("memory"))
        self.demand_time = int(dict.get("demand_time"))
        self.start_time = int(dict.get("start_time"))
        self.end_time = int(dict.get("end_time"))
        self.is_run = int(dict.get("is_run"))
        self.bid = float(dict.get("bid"))
        self.reserve = -1
        self.dict = dict

    def get_requirement_dict(self):
        return {"id": self.id,
                "cpu": self.cpu,
                "memory": self.memory,
                "demand_time": self.demand_time,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "is_run": self.is_run,
                "bid": self.bid,
                "reserve": self.reserve}

    def get_requirement_list(self):
        return [self.id, self.cpu, self.memory, self.demand_time,
                self.start_time, self.end_time, self.is_run,
                self.bid, self.reserve]

