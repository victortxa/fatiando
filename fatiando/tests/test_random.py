import numpy
from numpy.testing import assert_almost_equal
from fatiando import utils, gridder


def test_gridder_circular_scatter():
    "gridder.circular_scatter return diff sequence"
    area = [-1000, 1200, -40, 200]
    size = 1300
    for i in xrange(20):
        x1, y1 = gridder.circular_scatter(area, size, random=True)
        x2, y2 = gridder.circular_scatter(area, size, random=True)
        assert numpy.all(x1 != x2) and numpy.all(y1 != y2)


def test_gridder_circular_scatter_seed():
    "gridder.circular_scatter returns same sequence using same random seed"
    area = [0, 1000, 0, 1000]
    size = 1000
    for seed in numpy.random.randint(low=0, high=10000, size=20):
        x1, y1 = gridder.circular_scatter(area, size, random=True, seed=seed)
        x2, y2 = gridder.circular_scatter(area, size, random=True, seed=seed)
        assert numpy.all(x1 == x2) and numpy.all(y1 == y2)


def test_gridder_circular_scatter_seed_noseed():
    "gridder.circular_scatter returns diff sequence after using random seed"
    area = [0, 1000, 0, 1000]
    z = 20
    size = 1000
    seed = 1242
    x1, y1, z1 = gridder.circular_scatter(area, size, z,
                                          random=True, seed=seed)
    x2, y2, z2 = gridder.circular_scatter(area, size, z,
                                          random=True, seed=seed)
    assert numpy.all(x1 == x2) and numpy.all(y1 == y2)
    assert numpy.all(z1 == z2)
    x3, y3, z3 = gridder.circular_scatter(area, size, z, random=True)
    assert numpy.all(x1 != x3) and numpy.all(y1 != y3)
    assert numpy.all(z1 == z3)
    x4, y4, z4 = gridder.circular_scatter(area, size, z, random=False)
    assert numpy.all(x1 != x4) and numpy.all(y1 != y4)


def test_gridder_circular_scatter_constant():
    """
    gridder.circular_scatter must return points with the distance between 
    consecutive ones with a constant value, when ``random = False``.
    """
    area = [0, 1000, 0, 1000]
    size = 1000
    x, y = gridder.circular_scatter(area, size, random=False)
    for i in xrange(1, size-1):
        d1 = ((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)**0.5
        d2 = ((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)**0.5
        assert_almost_equal(d1, d2, 9)


def test_gridder_circular_scatter_num_point():
    "gridder.circular_scatter returns a specified ``n`` number of points."
    area = [0, 1000, 0, 1000]
    size = 1000
    x, y = gridder.circular_scatter(area, size, random=False)
    assert x.size == size and y.size == size
    x, y = gridder.circular_scatter(area, size, random=True)
    assert x.size == size and y.size == size


def test_gridder_scatter():
    "gridder.scatter returns diff sequence"
    area = [-1000, 1200, -40, 200]
    size = 1300
    for i in xrange(20):
        x1, y1 = gridder.scatter(area, size)
        x2, y2 = gridder.scatter(area, size)
        assert numpy.all(x1 != x2) and numpy.all(y1 != y2)


def test_gridder_scatter_seed():
    "gridder.scatter returns same sequence using same random seed"
    area = [0, 1000, 0, 1000]
    size = 1000
    for seed in numpy.random.randint(low=0, high=10000, size=20):
        x1, y1 = gridder.scatter(area, size, seed=seed)
        x2, y2 = gridder.scatter(area, size, seed=seed)
        assert numpy.all(x1 == x2) and numpy.all(y1 == y2)


def test_gridder_scatter_seed_noseed():
    "gridder.scatter returns diff sequence after using random seed"
    area = [0, 1000, 0, 1000]
    z = 20
    size = 1000
    seed = 1242
    x1, y1, z1 = gridder.scatter(area, size, z, seed=seed)
    x2, y2, z2 = gridder.scatter(area, size, z, seed=seed)
    assert numpy.all(x1 == x2) and numpy.all(y1 == y2)
    assert numpy.all(z1 == z2)
    x3, y3, z3 = gridder.scatter(area, size, z)
    assert numpy.all(x1 != x3) and numpy.all(y1 != y3)
    assert numpy.all(z1 == z3)


def test_gridder_scatter_num_point():
    "gridder.scatter returns a specified ``n`` number of points."
    area = [0, 1000, 0, 1000]
    size = 1000
    x, y = gridder.scatter(area, size)
    assert x.size == size and y.size == size


def test_utils_contaminate():
    "utils.contaminate generates noise with 0 mean and right stddev"
    size = 10 ** 6
    data = numpy.zeros(size)
    std = 4.213
    for i in xrange(20):
        noise = utils.contaminate(data, std)
        assert abs(noise.mean()) < 10 ** -10, 'mean:%g' % (noise.mean())
        assert abs(noise.std() - std) / std < 0.01, 'std:%g' % (noise.std())


def test_utils_contaminate_seed():
    "utils.contaminate noise with 0 mean and right stddev using random seed"
    size = 10 ** 6
    data = numpy.zeros(size)
    std = 4400.213
    for i in xrange(20):
        noise = utils.contaminate(data, std, seed=i)
        assert abs(noise.mean()) < 10 ** - \
            10, 's:%d mean:%g' % (i, noise.mean())
        assert abs(noise.std() - std) / std < 0.01, \
            's:%d std:%g' % (i, noise.std())


def test_utils_contaminate_diff():
    "utils.contaminate uses diff noise"
    size = 1235
    data = numpy.linspace(-100., 12255., size)
    noise = 244.4
    for i in xrange(20):
        d1 = utils.contaminate(data, noise)
        d2 = utils.contaminate(data, noise)
        assert numpy.all(d1 != d2)


def test_utils_contaminate_same_seed():
    "utils.contaminate uses same noise using same random seed"
    size = 1000
    data = numpy.linspace(-1000, 1000, size)
    noise = 10
    for seed in numpy.random.randint(low=0, high=10000, size=20):
        d1 = utils.contaminate(data, noise, seed=seed)
        d2 = utils.contaminate(data, noise, seed=seed)
        assert numpy.all(d1 == d2)


def test_utils_contaminate_seed_noseed():
    "utils.contaminate uses diff noise after using random seed"
    size = 1000
    data = numpy.linspace(-1000, 1000, size)
    noise = 10
    seed = 45212
    d1 = utils.contaminate(data, noise, seed=seed)
    d2 = utils.contaminate(data, noise, seed=seed)
    assert numpy.all(d1 == d2)
    d3 = utils.contaminate(data, noise)
    assert numpy.all(d1 != d3)
