import numpy as np  # Python 3.9.7, numpy version 1.23.1


class Simulation:
    def __init__(self, seed: int = None,
                 organization_size: int = 100, simulation_time: int = 60,
                 initial_metaknowledge_accuracy: float = 0.0, worker_dependency: float = 0.5,
                 initial_esm_adoption: float = 1.0, esm_interactivity: float = 0.5,
                 worker_interruption_tolerance: int = 60, worker_transparency_preference: int = 5,
                 metaknowledge_decay_rate: float = 0.0, worker_aspiration: float = 0.5,
                 **kwargs) -> None:

        if seed is not None:
            self.seed: int = seed
            np.random.seed(seed)

        # system parameters
        self.organization_size: int = organization_size  # number of workers
        # the maximum number of periods to run the model
        self.simulation_time: int = simulation_time
        # accuracy of how initial K captures I
        self.initial_metaknowledge_accuracy = initial_metaknowledge_accuracy
        # workplace configuration: density of the work dependency matrix
        self.worker_dependency: float = worker_dependency
        # ESM features
        self.esm_interactivity: float = esm_interactivity
        # initial percentage of workers using ESM
        self.initial_esm_adoption = initial_esm_adoption
        # workers attitude towards interruption and transparency
        self.worker_interruption_tolerance = worker_interruption_tolerance
        self.worker_transparency_preference = worker_transparency_preference
        # further parameters for robustness analysis
        self.metaknowledge_decay_rate = metaknowledge_decay_rate
        self.worker_aspiration = worker_aspiration
        # private parameters (_ not reportable)
        # need these matrices multiple times, save time and save it once
        self._zero: np.array = np.zeros(self.organization_size, np.int32)
        self._zeros: np.array = np.zeros(
            (self.organization_size, self.organization_size), np.int32)
        self._blank_slate: np.array = self._zeros - 1
        self._one: np.array = self._zero + 1
        self._ones: np.array = self._zeros + 1
        # matrices and vectors
        # the task matrix encoding workers' dependencies
        self._W: np.array = self.init_W()
        self._I: np.array = self._zeros.copy()  # ESM interaction
        # worker cost vector (number of interruptions)
        self._C: np.array = self._zero.copy()
        # K: metaknowledge matrix, represents knowledge of the task matrix W
        self._K: np.array = self.init_K()  # K: metaknowledge matrix
        self._M: np.array = self.init_M()  # M: ESM users vector
        self._T: np.array = self.init_T()  # T: transparency preference vector
        self._U: np.array = self.init_U()  # U: interruption tolerance vector
        # A: ESM users who posted, initiated with one (welcome) message
        self._A: np.array = self.select_k_ones(self._M, 1)
        # performance indicators, history is stored
        self.time_step_s: list = []
        self.adoption_s: list = []
        self.metaknowledge_accuracy_s: list = []
        self.realized_metaknowledge_s: list = []
        self.performance_s: list = []  # organizational performance
        self.interruptions_s: list = []
        self.leakiness_s: list = []
        # any extra parameters passed are stored
        for k, v in kwargs.items():
            exec(f'self.{k} = {v}')

    # helper functions
    def remove_diagonal(self, matrix) -> np.array:
        """zeros diagonals on the matrix since we don't count self-interaction"""
        mask = 1 - np.eye(self.organization_size, dtype=np.int32)
        return mask * matrix

    def select_with_probability(self, matrix, probability):
        "select with probability elements of value 1 from a matrix of 0 & 1"
        assert(matrix.ndim in [1, 2])
        assert(set(np.unique(matrix)).issubset({0, 1}))
        res = matrix.copy()
        if matrix.ndim == 1:
            x = np.where(res == 1)[0]
            replace_v = np.random.choice(
                [0, 1], len(x), p=[1-probability, probability])
            res[x] = replace_v
        elif matrix.ndim == 2:
            x, y = np.where(res == 1)
            replace_v = np.random.choice(
                [0, 1], len(x), p=[1-probability, probability])
            res[x, y] = replace_v
        return res

    def a_vector_of_n_ones_in_m_zeros(self, n, m, random=True):
        v = np.concatenate([np.ones(n), np.zeros(m - n)])
        if random:
            np.random.shuffle(v)
        return v

    def select_k_ones(self, vector, k, random=True):
        "(randomly) select k elements of value 1 from a vector of 0 & 1"
        assert(vector.ndim in [1])
        assert(set(np.unique(vector)).issubset({0, 1}))
        assert(k <= len(vector))
        # find the index of the one values
        x = np.where(vector == 1)[0]
        if random:
            np.random.shuffle(x)
        # select the first k values
        x = x[:k]  # index of the one values
        new_v = np.zeros(len(vector))
        new_v[x] = 1
        return new_v
    # end of helper functions

    def init_W(self) -> np.array:
        """Initialize worker dependency based on density"""
        # self-dependencies are meaningless, hence remove
        all_possible_dependencies = self.remove_diagonal(self._ones)
        w = self.select_with_probability(
            all_possible_dependencies, self.worker_dependency)
        return w

    def init_K(self) -> np.array:
        # initialize K based on initial_metaknowledge_accuracy
        # in K: -1 encodes no knowledge, 0 knowledge of lack of interdependence, 1 knowledge of interdependence
        all_possible_metaknowledge = self.remove_diagonal(self._ones)
        mask = self.select_with_probability(
            all_possible_metaknowledge, self.initial_metaknowledge_accuracy)
        w = self._W
        # here is the table the formula below implements
        # |mask | W   | K   |
        # | --- | --- | --- |
        # | 0   |  0  | -1  | when mask is 0 no knowledge
        # | 0   |  1  | -1  |
        # | 1   |  0  |  0  | when mask is 1 true knowledge of w
        # | 1   |  1  |  1  |
        k = -1 * (mask == 0) + ((mask == 1) & w)
        return self.remove_diagonal(k)

    def init_M(self) -> np.array:
        """Initialize the adoption of ESM based on initial ESM adoption"""
        m = self.select_with_probability(self._one, self.initial_esm_adoption)
        return m

    def init_T(self) -> np.array:
        t = np.random.poisson(
            lam=self.worker_transparency_preference, size=self.organization_size)
        return t

    def init_U(self) -> np.array:
        u = np.random.poisson(
            lam=self.worker_interruption_tolerance, size=self.organization_size)
        return u

    def update_metaknowledge(self, new_knowledge):  # has side effects!
        # knowledge overrides uncertainty (-1) and new knowledge overrides old knowledge
        old_knowledge = self._K.copy()
        x0, y0 = np.where(new_knowledge == 0)
        old_knowledge[x0, y0] = 0
        x1, y1 = np.where(new_knowledge == 1)
        old_knowledge[x1, y1] = 1
        self._K = self.remove_diagonal(old_knowledge)

    def learn_from_interacting(self, interactions: np.array):  # has side effects!
        # when interacting, workers uncover their true dependencies
        uncertainty = interactions + self._blank_slate  # -1 only for non selected workers
        knowledge = interactions & self._W  # the true w for selected workers
        new_knowledge = uncertainty + knowledge  # put -1 for non-selected workers
        self.update_metaknowledge(new_knowledge)

    def learn_from_observing_others_interactions(self, who: np.array, others: np.array):
        # 'who' learn about 'others'
        new_knowledge = self.remove_diagonal(
            self._blank_slate)  # start with a blank slate
        c = self._zero.copy()
        # index of 'who' revealed themselves
        revealing_others = np.where(others == 1)[0]
        learning_who = np.where(who == 1)[0]  # for instance through using ESM
        # assume observing these ESM interaction reveals information that helps
        # the recipient assesses their dependencies on them
        if len(revealing_others) * len(learning_who) > 0:  # both are non-empty
            new_knowledge[np.ix_(learning_who, revealing_others)] = self._W[np.ix_(
                learning_who, revealing_others)]
            c[learning_who] += self._ones[np.ix_(
                learning_who, revealing_others)].sum(axis=1)
        self._C = c
        self.update_metaknowledge(new_knowledge)

    def esm_interaction(self):  # has side effects!
        # step 1: interact with workers who posted on ESM
        potential_interactions = self._zeros.copy()
        source = np.where(self._M == 1)[0]
        target = np.where(self._A == 1)[0]
        potential_interactions[np.ix_(source, target)] = 1
        # step 2: realize only interactions where K is 1
        # assuming that workers are using ESM to promote knowledge work
        useful_interactions = potential_interactions * (self._K == 1)
        # step 3: each worker select up to its _T value from potential interactions
        interactions = np.array([self.select_k_ones(useful_interactions[i], self._T[i])
                                 for i in range(self.organization_size)], dtype='int32')
        # aspired workers interact
        self._I = self.select_with_probability(
            interactions, self.worker_aspiration)
        # step 4: cost resulting from interaction (resulting from interruption)
        # given the source interacted on purpose, the target receives the burden
        c = self._I.transpose()
        self._C += c.sum(axis=1)
        # step 5: record interacting workers (on the wall) for next period
        interacting_workers = self._I.sum(axis=1) != 0
        self._A = interacting_workers

    def observe_esm_interactions(self):
        who = self._M  # ESM users observe others' interaction
        # the observed interactions depend on ESM interactivity
        # and those who interacted recently
        others = self.select_with_probability(self._A, self.esm_interactivity)
        # update metaknowledge about others
        self.learn_from_observing_others_interactions(who, others)

    def change_use_behavior(self):
        problematic_workers = 0 + (self._C > self._U)
        # problematic_workers switch their behavior
        m = np.abs(self._M - problematic_workers)
        self._M = m
        self._C *= 0  # reset cost

    def forget(self):
        # forgetting: metaknowledge_decay_rate% of old metaknowledge is forgotten (switch to -1)
        forget_about = self.select_with_probability(
            self.remove_diagonal(self._ones), self.metaknowledge_decay_rate)
        x1, y1 = np.where(forget_about == 1)
        self._K[x1, y1] = -1

    def report(self, timestep):
        self.time_step_s.append(timestep)
        # update the actual adoption of ESM
        esm_adoption = self._M.mean()
        self.adoption_s.append(esm_adoption)
        # and the new metaknowledge_accuracy
        metaknowledge_accuracy = ((self._K == self._W).sum(
        ) - self.organization_size) / (self.organization_size - 1)**2
        self.metaknowledge_accuracy_s.append(metaknowledge_accuracy)
        realized_metaknowledge = self._I.sum() / ((self._K == 1).sum())
        self.realized_metaknowledge_s.append(realized_metaknowledge)
        # Organizational performance from ESM is the extent to which
        # the socialization it enables supports knowledge work
        performance = (self._I * self._W).sum() / self._W.sum()
        self.performance_s.append(performance)
        self.interruptions_s.append(self._C.sum() / self._M.sum())
        self.leakiness_s.append(self._A.sum() / self._M.sum())

    def step(self, timestep):
        self.change_use_behavior()  # consider behavior change based on last period
        # observe ESM interactions of last period (and learn new K)
        self.observe_esm_interactions()
        self.esm_interaction()  # initiate new interactions (and learn new K)
        self.forget()  # forget some of K
        self.report(timestep)  # report results

    def go(self, verbose=True):
        self.report(0)  # save initial values
        for t in range(self.simulation_time):
            if verbose:
                print(
                    f"t={t}\tP={self.performance_s[-1]:.2f}\tK={self.metaknowledge_accuracy_s[-1]:.2f}\tM={self.adoption_s[-1]:.2f}\tA={self._A.mean():.2f}\tC={self._C.mean():.2f}\tI={self._I.mean():.2f}")
            self.step(t+1)  # values after step t + 1


# one run to test code, to run multiple times check code in utilities.py
if __name__ == '__main__':
    s = Simulation()
    s.go(verbose=True)
