from make_test_data import DataGenerator

from svdrom.dmd import OptDMD

generator = DataGenerator()
generator.generate_svd_results(n_components=10)


def test_basic():
    optdmd = OptDMD()
    optdmd.fit(
        generator.u,
        generator.s,
        generator.v,
        varpro_opts_dict={"maxiter": 3},
    )
