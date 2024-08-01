import random
import math
import numpy as np

random.seed(0)


def logistic_map(mu, x):
    return mu * x * (1 - x)


def fitness(individual):
    fitness = 0
    for i in individual:
        a = i ** 2
        fitness += a
    return fitness


class i_GWO:
    def __init__(self, iterations, wolf_size, vector_size, min_range, max_range):
        """

        :param iterations: 最大迭代次数
        :param wolf_size: 灰狼个数
        :param vector_size: 灰狼个体维度
        :param min_range: 最小边界
        :param max_range: 最大边界
        """
        self.iterations = iterations
        self.wolf_size = wolf_size
        self.vector_size = vector_size
        self.min_range = min_range
        self.max_range = max_range

    # generate wolf pack
    def gen_wolf_pack(self):
        """
        初始化灰狼种群
        """
        min_range, max_range = self.min_range, self.max_range
        # 边界
        search_space = np.vstack((min_range, max_range))
        # print(search_space)
        # 种群初始化
        size = self.vector_size
        N = self.wolf_size
        population = np.zeros((N, size))
        x_init = np.random.rand(N, size)  # Random initialization in (0, 1)
        for i in range(N):
            x = x_init[i]
            for d in range(size):
                for _ in range(10):  # Iterations for logistic map to mix initial values
                    x[d] = logistic_map(3.9, x[d])
                population[i, d] = search_space[0, d] + x[d] * (search_space[1, d] - search_space[0, d])

        return population

    def hunt(self):
        # generate wolf pack
        wolf_pack = self.gen_wolf_pack()
        # sort pack by fitness
        pack_fit = sorted([(fitness(i), i) for i in wolf_pack])

        # main loop
        for k in range(self.iterations):
            # select alpha, beta and delta
            alpha, beta, delta = pack_fit[0][1], pack_fit[1][1], pack_fit[2][1]
            print('iteration: {}, best_wolf_position: {}'.format(k, fitness(alpha)))

            # linearly decreased from 2 to 0
            # a = 2*(1 - k/self.iterations)
            a = 2 / (1 + math.exp((20 * k / self.iterations) - 10))

            # updating each population member with the help of alpha, beta and delta
            for i in range(self.wolf_size):
                # compute A and C 
                A1, A2, A3 = a * (2 * random.random() - 1), a * (2 * random.random() - 1), a * (2 * random.random() - 1)
                C1, C2, C3 = 2 * random.random(), 2 * random.random(), 2 * random.random()

                # generate vectors for new position
                X1 = np.array([0.0 for i in range(self.vector_size)])
                X2 = np.array([0.0 for i in range(self.vector_size)])
                X3 = np.array([0.0 for i in range(self.vector_size)])

                # hunting
                for j in range(self.vector_size):
                    X1[j] = alpha[j] - A1 * abs(C1 * alpha[j] - wolf_pack[i][j])
                    X2[j] = beta[j] - A2 * abs(C2 * beta[j] - wolf_pack[i][j])
                    X3[j] = delta[j] - A3 * abs(C3 * delta[j] - wolf_pack[i][j])

                X_GWO = (X1 + X2 + X3) / 3.0

                # # 改进的自适应位置更新
                # fitness = self.fitness(wolf_pack[i])
                # fitness_values = [entry[0] for entry in pack_fit]
                # mean_fitness = sum(fitness_values) / len(fitness_values)
                # if fitness>=mean_fitness:
                #     for j in range(self.vector_size):
                #         X1[j] = alpha[j] - A1 * abs(C1 * alpha[j] - wolf_pack[i][j])
                #         X2[j] = beta[j] - A2 * abs(C2 * beta[j] - wolf_pack[i][j])
                #         X3[j] = delta[j] - A3 * abs(C3 * delta[j] - wolf_pack[i][j])
                #     X_GWO=(X1+X2+X3)/3.0
                # else:
                #     for j in range(self.vector_size):
                #         X1[j] = alpha[j] - A1 * abs(C1 * alpha[j] - wolf_pack[i][j])
                #         X2[j] = beta[j] - A2 * abs(C2 * beta[j] - wolf_pack[i][j])
                #         X3[j] = delta[j] - A3 * abs(C3 * delta[j] - wolf_pack[i][j])
                #     X_GWO = (pack_fit[0][0]*X1 + pack_fit[1][0]*X2 + pack_fit[2][0]*X3)/(pack_fit[0][0]+pack_fit[1][0]+pack_fit[2][0])

                # 基于维度学习的搜索策略
                # fitness calculation of new position candidate
                new_fitness = fitness(X_GWO)

                # current wolf fitness
                current_wolf = wolf_pack[i]

                # Begin i-GWO ehancement, Compute R --------------------------------
                R = fitness(current_wolf) - new_fitness

                # Compute eq. 11, build the neighborhood
                neighborhood = []
                for l in wolf_pack:
                    neighbor_distance = fitness(current_wolf) - fitness(l)
                    if neighbor_distance <= R:
                        neighborhood.append(l)

                # if the neigborhood is empy, compute the distance with respect 
                # to the other wolfs in the population and choose the one closer
                closer_neighbors = []
                if len(neighborhood) == 0:
                    for n in wolf_pack:
                        distance_wolf_alone = fitness(current_wolf) - fitness(n)
                        closer_neighbors.append((distance_wolf_alone, n))

                    closer_neighbors = sorted(closer_neighbors)
                    neighborhood.append(closer_neighbors[0][1])

                # Compute eq. 12 compute new candidate using neighborhood
                X_DLH = [0.0 for i in range(self.vector_size)]
                for m in range(self.vector_size):
                    random_neighbor = random.choice(neighborhood)
                    random_wolf_pop = random.choice(wolf_pack)

                    X_DLH[m] = current_wolf[m] + random.random() * random_neighbor[m] - random_wolf_pop[m]

                # if X_GWO is better than X_DLH, select X_DLH
                if fitness(X_GWO) < fitness(X_DLH):
                    candidate = X_GWO
                else:
                    candidate = X_DLH

                # if new position is better then replace, greedy update
                if fitness(candidate) < fitness(wolf_pack[i]):
                    wolf_pack[i] = candidate

            # sort the new positions by their fitness
            pack_fit = sorted([(fitness(i), i) for i in wolf_pack])


def main():
    model = i_GWO(iterations=15, wolf_size=10, vector_size=3, min_range=np.array([-1, -1, -1]),
                  max_range=np.array([1, 1, 1]))
    model.hunt()


if __name__ == '__main__':
    main()
