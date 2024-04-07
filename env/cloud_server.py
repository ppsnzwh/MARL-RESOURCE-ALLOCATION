import numpy as np
from unicodedata import decimal
from decimal import *
import random
class Cloud:
    def __init__(self, args):
        # 服务器id
        self.id = args.id
        self.cpu = args.cpu
        self.default_cpu = args.cpu
        self.cpu_price = args.cpu_price
        self.memory = args.memory
        self.default_memory = args.memory
        self.memory_price = args.memory_price
        self.run_requirement_list = []
        self.run_requirement_history_list = []
        self.time_step = 0
        self.cpu_use_ratio_history = []
        self.memory_use_ratio_history = []

    def compute_price(self, requirement):
        r = round(random.uniform(0.9, 1.1), 2)

        cpu_ratio = float((Decimal(str(self.default_cpu))-Decimal(str(self.cpu))) / Decimal(str(self.default_cpu)))
        memory_ratio = float((Decimal(str(self.default_memory)) - Decimal(str(self.memory))) / Decimal(str(self.default_memory)))

        #  资源利用率越高，出价越高，因为能耗在疯狂上升，简单线性提升，可以改进非线性
        cpu_price = (self.cpu_price * requirement.cpu) * (0.8+(cpu_ratio * 0.2))
        memory_price = (self.memory_price * requirement.memory) * (0.8+(memory_ratio * 0.2))

        result = round((float(Decimal(str(cpu_price))) + float(Decimal(str(memory_price))))  * float(Decimal(str(r))), 2)
        return result

    def compute_resource_ratio(self):
        cpu_ratio = float((Decimal(str(self.default_cpu))-Decimal(str(self.cpu))) / Decimal(str(self.default_cpu)))
        memory_ratio = float((Decimal(str(self.default_memory)) - Decimal(str(self.memory))) / Decimal(str(self.default_memory)))
        self.cpu_use_ratio_history.append(cpu_ratio)
        self.memory_use_ratio_history.append(memory_ratio)

    def execute_requirement(self, requirement):
        temp_cpu = self.cpu
        temp_memory = self.memory
        self.cpu = self.cpu - requirement.cpu
        self.memory = self.memory - requirement.memory
        if self.cpu < 0 or self.memory < 0:
            self.cpu = temp_cpu
            self.memory = temp_memory
            return False
        requirement.start_time = self.time_step
        requirement.is_run = 1
        requirement.reserve = self.id
        self.run_requirement_list.append(requirement)
        return True

    def release_requirement(self):
        run_req_num = len(self.run_requirement_list)
        if run_req_num <= 0:
            return 0

        release_index_list = []
        for i, j in zip(self.run_requirement_list, range(len(self.run_requirement_list))):
            req_start_time = i.start_time
            req_demand_time = i.demand_time
            if (self.time_step - req_start_time) >= req_demand_time:
                release_index_list.append(j)
        release_index_list.sort(reverse=True)
        for k in release_index_list:
            req = self.run_requirement_list[k]
            cpu = req.cpu
            memory = req.memory
            req.end_time = self.time_step
            self.cpu += cpu
            self.memory += memory
            self.run_requirement_list.pop(k)
            self.run_requirement_history_list.append(req)

    def get_obs(self, requirement):
        return [self.cpu, self.memory, requirement.cpu, requirement.memory]

    def show_args(self):
        return {"id": self.id,
                "cpu": self.cpu,
                "cpu_price": self.cpu_price,
                "memory": self.memory,
                "memory_price": self.memory_price}

    def avail_action(self, requirement, epsilon):
        if self.if_exec(requirement):
            return [1, 1]
        else:
            return [0, 1]

    def greed_avail_action(self, requirement):
        if self.if_exec(requirement):
            return 0
        else:
            return 1

    def if_exec(self, requirement):
        if self.cpu > requirement.cpu and self.memory > requirement.cpu:
            return True
        else:
            return False

    def avail_action_by_resource(self, requirement):
        if self.if_exec(requirement):
            epision = 0.1
            if epision < self.cpu/self.default_cpu or epision < self.cpu/self.default_cpu:
                return [1, 0]
            else:
                return [1, 1]
        else:
            return [0, 1]

    def reset(self):
        self.id = 0
        self.cpu = 0
        self.cpu_price = 0
        self.memory = 0
        self.memory_price = 0
        self.default_cpu = 0
        self.default_memory = 0
        self.run_requirement_list = []
        self.run_requirement_history_list = []
        self.time_step = 0
        self.cpu_use_ratio_history = []
        self.memory_use_ratio_history = []
