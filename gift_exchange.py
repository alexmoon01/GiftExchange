import math
from scipy.special import comb
from copy import deepcopy
import itertools
from matplotlib import pyplot as plt

def part(n, k, prev_parts=None):
    """
    Returns deduplicated list of all ways to write m as the sum of j integers
    """
    if prev_parts is None:
        prev_parts = {}
    if n < k or k < 1:
        raise Exception("Invalid partition args")
    if k == 1:
        return [[n]]
    if n == k:
        return [[1 for i in range(n)]]
    parts = []
    for i in range(math.ceil(float(n) / float(k)), n - k + 2):
        others = deepcopy(prev_parts.get((n - i, k - 1), part(n - i, k - 1, prev_parts)))
        for other in others:
            other.append(i)
        parts.extend(others)
    deduplicated = set(tuple(sorted(x)) for x in parts)
    uniq_parts = []
    for dedup in deduplicated:
        uniq_parts.append(list(dedup))
    if (n, k) not in prev_parts:
        prev_parts[(n, k)] = uniq_parts
    return uniq_parts


def f_exact(n, k):
    """
    Calculator for the amount of ways to arrange n nodes into k cycles
    :param n: The number of nodes
    :param k: The number of cycles
    :return: The amount of ways to arrange n nodes into k cycles
    """
    def fact(m):
        return math.factorial(m)

    partition = part(n, k)

    total = 0
    for p in partition:
        product = 1
        nodes_left = n
        counts = dict([(x, len(list(y))) for x, y in itertools.groupby(p)])
        for num in p:
            product *= fact(num - 1) * comb(nodes_left, num)
            nodes_left -= num
        for num in counts:
            product /= fact(counts[num])

        total += product
    return int(total)


def all_f(n, f=f_exact):
    """
    Returns all f from 1 to n
    :param n: The max num to consider
    :return: The dictionary containing data for all f from 1 to n
    """
    count = {}
    for k in range(n + 1)[1:]:
        count[(n, k)] = f(n, k)
    return count




if __name__ == "__main__":
    n = 40
    averages = []

    #for i in range(n + 1)[1:]:
    all_conf = all_f(n, f_exact)
    #    total = 0
    #    weighted = 0
    #    for k in all_conf:
    #        total += all_conf[k]
    #        weighted += k[1] * all_conf[k]
    #    averages.append(float(weighted)/float(total))
    #    print(i, averages[-1])
    #
    #print(averages)
    #plt.plot(range(n + 1)[1:], averages)
    #plt.xlabel('n')
    #plt.ylabel('Average number of cycles')
    #plt.show()

    ideal_total = math.factorial(n)
    test_total = sum(all_conf.values())
    print(ideal_total)
    print(test_total)
    print(float(abs(test_total - ideal_total))/float(ideal_total))
    plt.plot(range(n + 1)[1:], all_conf.values())
    plt.xlabel(f'k')
    plt.ylabel(f'Num Arrangements of {n} People into k Cycles')
    plt.show()

