from .gp_jax import rbf_kernel_jax
from .gp_jax import generate_gaussian_process as generate_gaussian_process_jax

from .utils_datagenNLE import init_multi

import torch
import numpy as np
import scipy.special
from logging import getLogger
from scipy.integrate import solve_ivp
from jax import numpy as jnp
from .data_gen_NLE import  diff_react_1D_f, burgers_f

from .fplanck import fokker_planck, boundary, gaussian_pdf, delta_function, uniform_pdf

from matplotlib import pyplot as plt
import seaborn as sns

logger = getLogger()

from symbolicregression.envs.node_utils import Node, NodeList
from symbolicregression.envs.ode_generator import ODEGenerator


class PDEGenerator(ODEGenerator):
    def __init__(self, params, float_encoder, equation_encoder, t_span, t_eval, x_grid, dt, dx):
        super().__init__(params, float_encoder, equation_encoder, t_span, t_eval, x_grid, dt, dx)

        if self.params.types == "pdebench":
            self.types = [
                # "advection",
                "diff_logisreact_1D",
                "burgers",
                "conservation_sinflux",
                "conservation_linearflux",
                # "compressiveNS",
            ]
        elif self.params.types == "pde_mol":
            self.types = ["heat", "twobody_diff_logisreact_1D", "kdv", "cahnhilliard_1D"]
        elif self.params.types == "pde":
            self.types = [
                "heat",
                "porous_medium",
                "advection",
                "kdv",
                "fplanck",
                "diff_logisreact_1D",
                "diff_linearreact_1D",
                "diff_bistablereact_1D",
                "diff_squarelogisticreact_1D",
                "burgers",
                "conservation_linearflux",
                "conservation_sinflux",
                "conservation_cubicflux",
                "inviscid_burgers",
                "inviscid_conservation_sinflux",
                "inviscid_conservation_cubicflux",
                "cahnhilliard_1D",
                "wave",
                "Klein_Gordon",
                "Sine_Gordon",
                # "compressiveNS",
                # "twobody_diff_logisreact_1D",
            ]
        elif self.params.types == "pde_burgers_type":
            self.types = [
                # "heat",
                "burgers",
                "conservation_sinflux",
                "conservation_cubicflux",
                "inviscid_burgers",
                "inviscid_conservation_sinflux",
                "inviscid_conservation_cubicflux",
            ]
        elif self.params.types == "pde_burgers_type2":
            self.types = [
                # "heat",
                "burgers",
                "conservation_cubicflux",
                "inviscid_burgers",
                "inviscid_conservation_cubicflux",
            ]
        elif self.params.types == "pde_burgers_type3":
            self.types = [
                "burgers",
                "conservation_cosflux",
                "conservation_cubicflux",
                "inviscid_burgers",
                "inviscid_conservation_cosflux",
                "inviscid_conservation_cubicflux",
            ]
        elif self.params.types == "pde_extr_train1":
            self.types = [
                "heat",
                "wave",
                "diff_linearreact_1D",
                "Sine_Gordon",
            ]
        elif self.params.types == "pde_extr_eval1":
            self.types = [
                "Klein_Gordon",
            ]
        elif self.params.types == "pde_extr1":
            self.types = [
                "heat",
                "wave",
                "diff_linearreact_1D",
                "Sine_Gordon",
                "Klein_Gordon"
            ]
        elif self.params.types == "pde_extr_train2":
            self.types = [
                "heat",
                "inviscid_burgers",
                "inviscid_conservation_sinflux",
                "inviscid_conservation_cubicflux",
                "conservation_sinflux",
                "conservation_cubicflux",
            ]
        elif self.params.types == "pde_extr_eval2":
            self.types = [
                "burgers",
            ]
        elif self.params.types == "pde_extr3":
            self.types = [
                "heat",
                "advection",
                "wave",
                "diff_logisreact_1D",
                "diff_linearreact_1D",
                "diff_bistablereact_1D",
                "diff_squarelogisticreact_1D",
                "burgers",
                "conservation_linearflux",
                "conservation_sinflux",
                "conservation_cubicflux",
                "inviscid_burgers",
                "inviscid_conservation_sinflux",
                "inviscid_conservation_cubicflux",
                "Klein_Gordon",
                "Sine_Gordon",
            ]
        elif self.params.types == "pde_extr_train3":
            self.types = [
                "diff_logisreact_1D",
                "diff_linearreact_1D",
                "diff_bistablereact_1D",
                "diff_squarelogisticreact_1D",
                "burgers",
                "conservation_linearflux",
                "conservation_sinflux",
                "conservation_cubicflux",
                "inviscid_burgers",
                "inviscid_conservation_sinflux",
                "inviscid_conservation_cubicflux",
                "Klein_Gordon",
                "Sine_Gordon",
            ]
        elif self.params.types == "pde_extr_eval3":
            self.types = [
                "heat",
                "advection",
                "wave",
            ]
        elif self.params.types == "pde_extr4":
            self.types = [
                "heat",
                "diff_logisreact_1D",
                "diff_linearreact_1D",
                "diff_bistablereact_1D",
                "diff_squarelogisticreact_1D",
            ]
        elif self.params.types == "pde_extr_train4":
            self.types = [
                "diff_logisreact_1D",
                "diff_linearreact_1D",
                "diff_bistablereact_1D",
                "diff_squarelogisticreact_1D",
            ]
        elif self.params.types == "pde_extr_eval4":
            self.types = [
                "heat",
            ]
        elif self.params.types == "pde_sameIC":
            self.types = [
                "heat",
                "kdv",
                "diff_logisreact_1D",
                "diff_linearreact_1D",
                "diff_bistablereact_1D",
                "diff_squarelogisticreact_1D",
                "burgers",
                "conservation_linearflux",
                "conservation_sinflux",
                "conservation_cubicflux",
                "inviscid_burgers",
                "inviscid_conservation_sinflux",
                "inviscid_conservation_cubicflux",
                "cahnhilliard_1D",
                "Klein_Gordon",
                "Sine_Gordon",
                # "compressiveNS",
                # "twobody_diff_logisreact_1D",
            ]

        else:
            try:
                self.types = self.params.types.split(",")
                self.types = [
                    pde_type.split("pde_")[-1] for pde_type in self.types if pde_type.startswith("pde_")
                ]
                assert len(self.types) > 0
            except:
                assert False, "invalid type: {}".format(self.params.types)
        # if self.params.eval_types == "pde_burgers_type":
        #     self.eval_types = [
        #         "heat",
        #         "burgers",
        #         "conservation_sinflux",
        #         "conservation_cubicflux",
        #         "inviscid_burgers",
        #         "inviscid_conservation_sinflux",
        #         "inviscid_conservation_cubicflux",
        #     ]
        # else:
        #     try:
        #         self.eval_types = self.params.eval_types.split(",")
        #         self.eval_types = [
        #                 pde_type.split("pde_")[-1] for pde_type in self.eval_types if pde_type.startswith("pde_")
        #             ]
        #     except:
        #         assert False, "invalid type: {}".format(self.params.eval_types)

        self.tfinals = {
            "heat" : self.params.t_range,
            "porous_medium": 0.1,
            "advection":self.params.t_range,
            "kdv":1,
            "fplanck":.1,
            "diff_logisreact_1D":self.params.t_range,
            "diff_linearreact_1D":self.params.t_range,
            "diff_bistablereact_1D":self.params.t_range,
            "diff_squarelogisticreact_1D":self.params.t_range,
            "burgers":self.params.t_range,
            "conservation_linearflux":self.params.t_range,
            "conservation_sinflux":self.params.t_range,
            "conservation_cosflux": self.params.t_range,
            "conservation_cubicflux":self.params.t_range,
            "inviscid_burgers":self.params.t_range,
            "inviscid_conservation_sinflux":self.params.t_range,
            "inviscid_conservation_cosflux": self.params.t_range,
            "inviscid_conservation_cubicflux":self.params.t_range,
            "cahnhilliard_1D":.5,
            "wave":1,
            "Klein_Gordon":1,
            "Sine_Gordon":1,
        }
        if self.params.t_final != "":
            change_t_final =  self.params.t_final.split(";")
            for _change_t_final in change_t_final:
                cur_type,cur_t_final = _change_t_final.split(",")
                if cur_type == "burgers_type":
                    cur_type = ["burgers",
                "conservation_linearflux",
                "conservation_sinflux",
                "conservation_cubicflux",
                "inviscid_burgers",
                "inviscid_conservation_sinflux",
                "inviscid_conservation_cubicflux",]
                    for _cur_type in cur_type:
                        self.tfinals[_cur_type] = int(cur_t_final)
                elif cur_type == "diff_react":
                    cur_type = ["diff_logisreact_1D",
                "diff_linearreact_1D",
                "diff_bistablereact_1D",
                "diff_squarelogisticreact_1D",]
                    for _cur_type in cur_type:
                        self.tfinals[_cur_type] = int(cur_t_final)
                else:
                    self.tfinals[cur_type] = int(cur_t_final)
        self.type_to_dim = {
            "heat": 1,
            "advection": 1,
            "kdv": 1,
            "fplanck": 1,
            "cahnhilliard_1D": 1,
            "diff_logisreact_1D": 1,
            "diff_linearreact_1D": 1,
            "diff_bistablereact_1D": 1,
            "diff_squarelogisticreact_1D": 1,
            "burgers": 1,
            "conservation_linearflux": 1,
            "conservation_sinflux": 1,
            "conservation_cosflux": 1,
            "conservation_cubicflux": 1,
            "inviscid_burgers": 1,
            "inviscid_conservation_linearflux": 1,
            "inviscid_conservation_sinflux": 1,
            "inviscid_conservation_cosflux": 1,
            "inviscid_conservation_cubicflux": 1,
            "porous_medium": 1,
            "Klein_Gordon": 1,
            "Sine_Gordon": 1,
            "wave":1,
            "compressiveNS": 3,
            "twobody_diff_react_1D": 2,
        }

        self.cur_idx = 0
        self.total_types = len(self.types)
        self.termlist = ["u", "ut", "ux", "uxx", "uxxx", "uxxxx"]

        if self.params.noisy_text_input:
            p = self.params
            self.missing_locations = dict()
            self.addition_locations = dict()

            # generate terms to be added (polynomials of degree at most 2)
            self.addition_terms = dict()
            for dim in range(
                self.params.min_output_dimension, self.params.max_output_dimension + 1
            ):  # max_output_dimension = max_pde_mesh
                cur_addition_terms = [Node(self.ph, p)]

                for k in range(len(self.termlist)):
                    for i in range(dim):
                        cur_addition_terms.append(
                            Node("mul", p, [Node(self.ph, p), Node(f"{self.termlist[k]}_{i}", p)])
                        )

                        for j in range(i, dim):
                            for m in range(k, len(self.termlist)):
                                cur_addition_terms.append(
                                    Node(
                                        "mul",
                                        p,
                                        [
                                            Node(self.ph, p),
                                            Node(
                                                "mul",
                                                p,
                                                [
                                                    Node(f"{self.termlist[k]}_{i}", p),
                                                    Node(f"{self.termlist[m]}_{j}", p),
                                                ],
                                            ),
                                        ],
                                    )
                                )
                self.addition_terms[dim] = cur_addition_terms

        self.shared_ICs = None
        self.shared_coeff = None

    def generate_sample(self, rng, train=True, type=None):
        """
        Generate a tree sample
        """
        if type is None:
            if self.shared_ICs is None and self.params.use_sharedIC:
                num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
                # self.shared_ICs = None
                if self.params.IC_types == "train":
                    self.shared_ICs  =init_multi(
                            self.x_grid.flatten(),
                            numbers=num_initial_points * 10,
                            k_tot=2,
                            init_key=rng.randint(100000),
                            if_norm=True,
                            norm_const = self.params.IC_range,
                        )

                else:
                    self.shared_ICs  = generate_gaussian_process_jax(
                            self.x_grid.flatten(),
                            init_key=rng.randint(100000),
                            num=num_initial_points * 10,
                            kernel=rbf_kernel_jax,
                            k_sigma=1,
                            k_l=0.5,
                            norm_const=self.params.IC_range,
                        )
            if self.shared_coeff is None and self.params.use_sharedcoeff:
                if self.params.extrapolate_pdetypes:
                    self.shared_diff_coeff = .01
                else:
                    self.shared_diff_coeff = 3e-3
                c1_range = self.get_sample_range(self.shared_diff_coeff)
                self.shared_diff_coeff = self.refine_floats(rng.uniform(*c1_range, (1,)))[0]

                self.shared_flux_coeff = 1
                c1_range = self.get_sample_range(self.shared_flux_coeff)
                self.shared_flux_coeff = self.refine_floats(rng.uniform(*c1_range, (1,)))[0]

                self.shared_coeff = [self.shared_diff_coeff , self.shared_flux_coeff]


            type = self.types[self.cur_idx]

            self.cur_idx = self.cur_idx + 1
            if self.cur_idx >= self.total_types:
                self.cur_idx = 0
                if self.params.use_sharedIC:
                    num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
                    # self.shared_ICs = None
                    if self.params.IC_types == "train":
                        self.shared_ICs = init_multi(
                            self.x_grid.flatten(),
                            numbers=num_initial_points * 10,
                            k_tot=2,
                            init_key=rng.randint(100000),
                            if_norm=True,
                            norm_const=self.params.IC_range,
                        )

                    else:
                        self.shared_ICs = generate_gaussian_process_jax(
                            self.x_grid.flatten(),
                            init_key=rng.randint(100000),
                            num=num_initial_points * 10,
                            kernel=rbf_kernel_jax,
                            k_sigma=1,
                            k_l=0.2,
                            norm_const=self.params.IC_range,
                        )
                if self.params.use_sharedcoeff:
                    if self.params.extrapolate_pdetypes:
                        self.shared_diff_coeff = .01
                    else:
                        self.shared_diff_coeff = 3e-3
                    c1_range = self.get_sample_range(self.shared_diff_coeff)
                    self.shared_diff_coeff = self.refine_floats(rng.uniform(*c1_range, (1,)))[0]

                    self.shared_flux_coeff = 1
                    c1_range = self.get_sample_range(self.shared_flux_coeff)
                    self.shared_flux_coeff = self.refine_floats(rng.uniform(*c1_range, (1,)))[0]

                    self.shared_coeff = [self.shared_diff_coeff, self.shared_flux_coeff]

        item = getattr(self, "generate_" + type)(rng, train, self.shared_ICs,self.shared_coeff)

        return item

    def heat_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["sub"]]
        term_list = [
            [
                self.mul_terms([ph, "ut_0"]),
                self.mul_terms([ph, "uxx_0"]),
            ]
        ]
        return op_list, term_list

    def generate_heat(self, rng, train, ICs=None,coeff = None):
        item = {"type": "heat"}
        p = self.params
        if coeff is not None:
            c1 = coeff[0]
        else:
            if p.extrapolate_pdetypes:
                c1 = .01
            else:
                c1 = 3e-3  # 2e-3
            c1_range = self.get_sample_range(c1)
            c1 = self.refine_floats(rng.uniform(*c1_range, (1,)))[0]

        tf = self.tfinals["heat"]
        coeff_t = p.t_range/tf

        op_list = [["sub"]]
        term_list = [
            [
                self.mul_terms([str(coeff_t), "ut_0"]),
                self.mul_terms([str(c1), "uxx_0"]),
            ]
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        #
        def f_closure(c1):

            def f(t, u):
                d2u_dx2 = np.zeros_like(u)
                dx = p.x_range / p.x_num
                # Compute second spatial derivatives using central differences
                for i in range(1, p.x_num - 1):
                    d2u_dx2[i] = (u[i - 1] - 2 * u[i] + u[i + 1]) / dx**2

                # Periodic boundary conditions
                d2u_dx2[0] = (u[-1] - 2 * u[0] + u[1]) / dx**2
                d2u_dx2[-1] = (u[-2] - 2 * u[-1] + u[0]) / dx**2

                du_dt = c1 * d2u_dx2
                return du_dt

            return f

        item["func"] = f_closure(c1)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        if ICs is not None:
            y_0s = np.array(ICs)
        elif p.IC_types == "train":
            y_0s = np.array(
                init_multi(
                    self.x_grid.flatten(),
                    numbers=num_initial_points * 10,
                    k_tot=4,
                    init_key=rng.randint(100000),
                    if_norm=True,
                )
            )
        else:
            y_0s = np.array(
                generate_gaussian_process_jax(
                    self.x_grid.flatten(),
                    init_key=rng.randint(100000),
                    num=num_initial_points * 10,
                    kernel=rbf_kernel_jax,
                    k_sigma=1,
                    k_l=0.2,
                )
            )
            # slope = (y_0s[:,-1] - y_0s[:,0])/p.x_range
            #
            # y_0s -= np.outer(slope,self.x_grid.flatten() )
            #
            # y_0s = (y_0s - np.min(y_0s,axis = 1).reshape(-1,1))/ (np.max(y_0s,axis = 1).reshape(-1,1)-np.min(y_0s,axis = 1).reshape(-1,1))
        res = []
        fun = item["func"]
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    [t /coeff_t for t in self.t_span],
                    y_0,
                    method="BDF",
                    t_eval=self.t_eval/coeff_t,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)).unsqueeze(-1))
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        #
        # for i in range(len(res)):
        #     sns.heatmap(np.asarray(res[i])[:, :, 0])
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title('Heat Equation with diffusion coefficient = ' + str(c1))
        #     plt.show()
        #
        #     plt.plot(res[i][0,:,0])
        #     plt.show()

        return item

    def porous_medium_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add", "add"]]
        term_list = [
            [
                self.mul_terms([ph, "ut_0"]),
                Node(
                    "mul",
                    p,
                    [
                        Node(ph, p),
                        Node(
                            "mul",
                            p,
                            [Node("pow", p, [Node("u_0", p), Node(ph, p)]), Node("pow2", p, [Node("ux_0", p)])],
                        ),
                    ],
                ),
                Node(
                    "mul",
                    p,
                    [Node(ph, p), Node("mul", p, [Node("pow", p, [Node("u_0", p), Node(ph, p)]), Node("uxx_0", p)])],
                ),
            ]
        ]
        return op_list, term_list

    def generate_porous_medium(self, rng, train, ICs=None,coeff = None):  # ignore ICs here

        item = {"type": "porous_medium"}

        m = rng.randint(2, 5)

        p = self.params
        tf = self.tfinals["porous_medium"]
        coeff = p.t_range / tf
        op_list = [["add", "add"]]
        term_list = [
            [
                self.mul_terms([str(coeff), "ut_0"]),
                Node(
                    "mul",
                    p,
                    [
                        Node(str(m * (m - 1)), p),
                        Node(
                            "mul",
                            p,
                            [Node("pow", p, [Node("u_0", p), Node(str(m - 2), p)]), Node("pow2", p, [Node("ux_0", p)])],
                        ),
                    ],
                ),
                Node(
                    "mul",
                    p,
                    [
                        Node(str(m), p),
                        Node("mul", p, [Node("pow", p, [Node("u_0", p), Node(str(m - 1), p)]), Node("uxx_0", p)]),
                    ],
                ),
            ]
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        #
        def f_closure(m):

            def f(t, u):
                d2um_dx2 = np.zeros_like(u)
                dx = p.x_range / p.x_num
                um = np.power(u, m)
                # Compute second spatial derivatives using central differences
                for i in range(1, p.x_num - 1):
                    d2um_dx2[i] = (um[i - 1] - 2 * um[i] + um[i + 1]) / dx**2

                # Periodic boundary conditions
                d2um_dx2[0] = (um[-1] - 2 * um[0] + um[1]) / dx**2
                d2um_dx2[-1] = (um[-2] - 2 * um[-1] + um[0]) / dx**2

                du_dt = d2um_dx2
                return du_dt

            return f

        item["func"] = f_closure(m)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        # y_0s = rng.uniform(-2, 2, (num_initial_points * 10, p.x_num))
        # y_0s = np.array(init_multi(self.x_grid.flatten(), numbers=num_initial_points*10, k_tot=4, init_key=rng.randint(100000)))
        res = []
        fun = item["func"]

        for i in range(num_initial_points * 10):

            if p.IC_types == "train":
                A_range = [9, 11]
                center_range = [0.9, 1.1]
                std_range = [0.9, 1.1]
                A = self.refine_floats(rng.uniform(*A_range, (1,)))[0]
                center = self.refine_floats(rng.uniform(*center_range, (1,)))[0]
                std = self.refine_floats(rng.uniform(*std_range, (1,)))[0]
                y_0 = np.exp(-A * ((self.x_grid.flatten() - center) ** 2) / (2 * std**2))
                slope = (y_0[-1] - y_0[0]) / p.x_range
                y_0 -= slope * self.x_grid.flatten()
                y_0 = (y_0 - np.min(y_0)) / (np.max(y_0) - np.min(y_0))
            else:
                A_range = [5, 15]
                center_range = [0.9, 1.1]
                std_range = [0.3, 0.5]
                A = self.refine_floats(rng.uniform(*A_range, (1,)))[0]
                center = self.refine_floats(rng.uniform(*center_range, (1,)))[0]
                std = self.refine_floats(rng.uniform(*std_range, (1,)))[0]
                y_0 = np.maximum(-A * (self.x_grid.flatten() - center) ** 2 / (2 * std**2) + A, 0)
                slope = (y_0[-1] - y_0[0]) / p.x_range
                y_0 -= slope * self.x_grid.flatten()
                y_0 = (y_0 - np.min(y_0)) / (np.max(y_0) - np.min(y_0))

            # y_0 = y_0s[i,:]
            try:
                sol = solve_ivp(
                    fun,
                    [t / coeff for t in self.t_span],
                    y_0,
                    method="RK45",
                    t_eval=self.t_eval / coeff,
                    rtol=self.rtol,
                    atol=self.atol,
                    max_step=0.0001,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)).unsqueeze(-1))
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res

        # for i in range(len(res)):
        #     this = res[i]
        #     sns.heatmap(np.asarray(res[i])[:, :, 0])
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title('porous_medium Equation ' )
        #     plt.show()
        #     for j in range(len(self.t_eval)):
        #         plt.plot(this[j,:])
        #     plt.show()
        #     plt.close()

        return item

    def Klein_Gordon_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["sub", "add"]]
        term_list = [[self.mul_terms([ph, "utt_0"]), self.mul_terms([ph, "uxx_0"]), self.mul_terms([ph, "u_0"])]]
        return op_list, term_list

    def generate_Klein_Gordon(self, rng, train, ICs=None,coeff = None):
        p = self.params
        c = 1
        if p.extrapolate_pdetypes:
            m = 0.35
        else:
            m = 0.1

        c_range = self.get_sample_range(c)
        m_range = self.get_sample_range(m)
        item = {"type": "Klein_Gordon"}

        c = self.refine_floats(rng.uniform(*c_range, (1,)))[0]
        m = self.refine_floats(rng.uniform(*m_range, (1,)))[0]


        tf = self.tfinals["Klein_Gordon"]
        coeff = p.t_range / tf

        op_list = [["sub", "add"]]
        term_list = [
            [
                self.mul_terms([str(coeff**2), "utt_0"]),
                self.mul_terms([str(c**2), "uxx_0"]),
                self.mul_terms([str(m**2 * c**4), "u_0"]),
            ]
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        dt_this = self.dt / (100 * coeff)
        alpha = (c * dt_this / self.dx) ** 2
        beta = m**2 * c**4 * dt_this**2

        def update(u, upr, t_curr, t_save):
            while t_curr < t_save:
                unew = 2 * (1 - alpha) * u - upr + alpha * (np.roll(u, -1) + np.roll(u, 1)) - beta * u  # t+1
                upr = u
                u = unew
                t_curr = t_curr + dt_this
            return u, upr, t_curr

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        res = []

        # if p.IC_types == "train":
        #     center_range = [.5, 1.5]
        #     std_range = [.2, .6]
        # else:
        #     center_range = [.1, 1.9]
        #     std_range = [.01, .9]
        if ICs is not None:
            y_0s = np.array(ICs)
        elif p.IC_types == "train":
            y_0s = np.array(
                init_multi(
                    self.x_grid.flatten(),
                    numbers=num_initial_points * 10,
                    k_tot=2,
                    num_choise_k=1,
                    init_key=rng.randint(100000),
                    if_norm=True,
                )
            )
        else:
            y_0s = np.array(
                generate_gaussian_process_jax(
                    self.x_grid.flatten(),
                    init_key=rng.randint(100000),
                    num=num_initial_points * 10,
                    kernel=rbf_kernel_jax,
                    k_sigma=1,
                    k_l=0.2,
                    if_norm=True,
                )
            )
        # slope = (y_0s[:,-1] - y_0s[:,0])/p.x_range
        #
        # y_0s -= np.outer(slope,self.x_grid.flatten() )
        #
        # y_0s = (y_0s - np.min(y_0s, axis=1).reshape(-1, 1)) / (
        #             np.max(y_0s, axis=1).reshape(-1, 1) - np.min(y_0s, axis=1).reshape(-1, 1))

        for i in range(num_initial_points * 10):

            psi_t0 = np.zeros(self.x_grid.flatten().shape)
            try:
                # center = self.refine_floats(rng.uniform(*center_range, (1,)))[0]
                # std = self.refine_floats(rng.uniform(*std_range, (1,)))[0]
                # psi_0 = np.exp(-((self.x_grid.flatten() - center ) ** 2) / (
                #             2 * std ** 2))
                # slope = (psi_0[-1] - psi_0[0])/p.x_range
                # psi_0 -= slope * self.x_grid.flatten()# Gaussian packet at t=0
                psi_0 = y_0s[i, :]
                y = [psi_0]
                upr = psi_0
                u = psi_0 + dt_this * psi_t0
                t_current = dt_this

                for n in range(1, p.t_num):
                    t_save = self.t_eval[n] / coeff
                    u, upr, t_current = update(u, upr, t_current, t_save)
                    y.append(u)
                y = np.array(y)
                res.append(torch.from_numpy(y.astype(np.single)).unsqueeze(-1))
                if len(res) >= num_initial_points:
                    break
            except Exception as e:
                pass
        item["data"] = res

        for i in range(len(res)):
            this_sample = np.asarray(res[i])[:, :, 0]
            sns.heatmap(this_sample)
            plt.xlabel('x')
            plt.ylabel('t')
            plt.title('KG Equation  ' )
            plt.show()
            plt.close()
            for j in range(p.t_num):
                if j % 5 ==0:
                    plt.plot(this_sample[j,:])
            plt.plot(this_sample[0, :],linewidth = 5)
            plt.ylim([0,1])
            plt.show()
            plt.close()

        return item

    def Sine_Gordon_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["sub", "add"]]
        term_list = [
            [
                self.mul_terms([ph, "utt_0"]),
                Node("uxx_0", p),
                Node("mul", p, [Node(ph, p), Node("sin", p, [Node("u_0", p)])]),
            ]
        ]
        return op_list, term_list

    def generate_Sine_Gordon(self, rng, train, ICs=None,coeff = None):

        item = {"type": "Sine_Gordon"}

        p = self.params
        tf = self.tfinals["Sine_Gordon"]
        coeff = p.t_range / tf
        if p.extrapolate_pdetypes:
            c = 3e-3
        else:
            c = 1
        c_range = self.get_sample_range(c)
        c = self.refine_floats(rng.uniform(*c_range, (1,)))[0]

        op_list = [["sub", "add"]]
        term_list = [
            [
                self.mul_terms([str(coeff**2), "utt_0"]),
                Node("uxx_0", p),
                Node("mul", p, [Node(str(c), p), Node("sin", p, [Node("u_0", p)])]),
            ]
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        dt_this = self.dt / (coeff * 100)

        def update(u, upr, t_curr, t_save):
            while t_curr < t_save:
                alpha = (dt_this**2) / (self.dx**2)
                u_new = 2 * u - upr + alpha * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) - (dt_this**2) * c * np.sin(u)
                upr = u
                u = u_new
                t_curr = t_curr + dt_this
            return u, upr, t_curr

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        res = []
        if ICs is not None:
            y_0s = np.array(ICs)
        elif p.IC_types == "train":
            y_0s = np.array(
                init_multi(
                    self.x_grid.flatten(),
                    numbers=num_initial_points * 10,
                    k_tot=2,
                    num_choise_k=1,
                    init_key=rng.randint(100000),
                    if_norm=True,
                )
            )
        else:
            y_0s = np.array(
                generate_gaussian_process_jax(
                    self.x_grid.flatten(),
                    init_key=rng.randint(100000),
                    num=num_initial_points * 10,
                    kernel=rbf_kernel_jax,
                    k_sigma=1,
                    k_l=0.2,
                    if_norm=True,
                )
            )

        for i in range(num_initial_points * 10):

            # center_range = [.2, 1.8]
            # center = self.refine_floats(rng.uniform(*center_range, (1,)))[0]
            # v_range = [.1, .9]
            # v = self.refine_floats(rng.uniform(*v_range, (1,)))[0]
            # gamma = 1 / np.sqrt(1 - v ** 2)
            # A_range = [.1,10]
            # A = self.refine_floats(rng.uniform(*A_range, (1,)))[0]
            # psi_0 = A* np.arctan(np.exp(gamma * (self.x_grid.flatten() - center)))
            # psi_t0 = np.zeros(self.x_grid.flatten() .shape)
            # x_grid = np.linspace(0,2,200)

            psi_t0 = np.zeros(self.x_grid.flatten().shape)
            try:
                # center = self.refine_floats(rng.uniform(*center_range, (1,)))[0]
                # std = self.refine_floats(rng.uniform(*std_range, (1,)))[0]
                # psi_0 = np.exp(-((self.x_grid.flatten() - center ) ** 2) / (
                #                 2 * std ** 2))  # Gaussian packet at t=0
                #
                # slope = (psi_0[-1] - psi_0[0])/p.x_range
                # psi_0 -= slope * self.x_grid.flatten()# Gaussian packet at t=0
                psi_0 = y_0s[i, :]
                y = [psi_0]
                upr = psi_0
                u = psi_0 + dt_this * psi_t0
                t_current = dt_this
                for n in range(1, p.t_num):
                    t_save = self.t_eval[n] / coeff
                    u, upr, t_current = update(u, upr, t_current, t_save)
                    y.append(u)
                y = np.array(y)
                res.append(torch.from_numpy(y.astype(np.single)).unsqueeze(-1))
                if len(res) >= num_initial_points:
                    break
            except Exception as e:
                pass
        item["data"] = res
        #
        # for i in range(len(res)):
        #     this_sample = np.asarray(res[i])[:, :, 0]
        #     sns.heatmap(this_sample)
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title('SG Equation  ' )
        #     plt.show()
        #     plt.close()
        #     for j in range(p.t_num):
        #         if j % 5 ==0:
        #             plt.plot(this_sample[j,:])
        #     plt.plot(this_sample[0,:],linewidth = 5)
        #     plt.ylim([0,1])
        #     plt.show()
        #     plt.close()

        return item

    def cahnhilliard_1D_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add", "add", "add"]]
        term_list = [
            [
                self.mul_terms([ph, "ut_0"]),
                self.mul_terms([ph, "uxxxx_0"]),
                self.mul_terms([ph, "ux_0", "ux_0"]),
                self.mul_terms([ph, "u_0", "uxx_0"]),
            ]
        ]
        return op_list, term_list

    def generate_cahnhilliard_1D(self, rng, train, ICs=None,coeff = None):
        eps = 0.01
        tf = self.tfinals["cahnhilliard_1D"]
        p = self.params
        coeff = p.t_range / tf
        eps_range = self.get_sample_range(eps)

        item = {"type": "cahnhilliard_1D"}

        eps = self.refine_floats(rng.uniform(*eps_range, (1,)))[0]

        op_list = [["add", "add", "add"]]
        term_list = [
            [
                self.mul_terms([str(coeff), "ut_0"]),
                self.mul_terms([str(eps**2), "uxxxx_0"]),
                self.mul_terms([str(6), "ux_0", "ux_0"]),
                self.mul_terms([str(6), "u_0", "uxx_0"]),
            ]
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        #
        def f_closure(eps):

            def f(t, u):
                d2u_dx2 = np.zeros_like(u)
                rhs = np.zeros_like(u)
                dx = p.x_range / p.x_num
                # Compute second spatial derivatives using central differences
                for i in range(1, p.x_num - 1):
                    d2u_dx2[i] = (u[i - 1] - 2 * u[i] + u[i + 1]) / dx**2

                # Periodic boundary conditions
                d2u_dx2[0] = (u[-1] - 2 * u[0] + u[1]) / dx**2
                d2u_dx2[-1] = (u[-2] - 2 * u[-1] + u[0]) / dx**2

                f = u**3 - u
                fu = 3 * u**2 - 1

                d2u_dx2af = eps**2 * d2u_dx2 + fu

                for i in range(1, p.x_num - 1):
                    rhs[i] = (d2u_dx2af[i - 1] - 2 * d2u_dx2af[i] + d2u_dx2af[i + 1]) / dx**2

                # Periodic boundary conditions
                rhs[0] = (d2u_dx2af[-1] - 2 * d2u_dx2af[0] + d2u_dx2af[1]) / dx**2
                rhs[-1] = (d2u_dx2af[-2] - 2 * d2u_dx2af[-1] + d2u_dx2af[0]) / dx**2

                du_dt = -d2u_dx2af
                return du_dt

            return f

        item["func"] = f_closure(eps)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        if ICs is not None:
            y_0s = np.array(ICs)
        elif p.IC_types == "train":
            y_0s = np.array(
                init_multi(
                    self.x_grid.flatten(),
                    numbers=num_initial_points * 10,
                    k_tot=2,
                    init_key=rng.randint(100000),
                    if_norm=True,
                )
            )
        else:
            y_0s = np.array(
                generate_gaussian_process_jax(
                    self.x_grid.flatten(),
                    init_key=rng.randint(100000),
                    num=num_initial_points * 10,
                    kernel=rbf_kernel_jax,
                    k_sigma=1,
                    k_l=0.2,
                )
            )
        res = []
        fun = item["func"]
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    [t / coeff for t in self.t_span],
                    y_0,
                    method="BDF",
                    t_eval=self.t_eval / coeff,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)).unsqueeze(-1))
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res

        # for i in range(len(res)):
        #
        #     this_sample = np.asarray(res[i])
        #     sns.heatmap(np.asarray(res[i])[:, :, 0])
        #     # print(np.linalg.norm(this_sample[32:,:,0]))
        #     print(np.linalg.norm(this_sample[32:,:,0])/(np.linalg.norm(this_sample[32:,:,0] - np.mean(this_sample[32:,:,0]))))
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title('CH 1D Equation with coefficient = ' + str(eps))
        #     plt.show()
        #     for j in range(p.t_num):
        #         plt.plot(this_sample[j,:])
        #     plt.plot(this_sample[0,:],linewidth = 5)
        #     plt.show()
        #     plt.close()

        return item

    def kdv_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add", "add"]]
        term_list = [[self.mul_terms([ph, "ut_0"]), self.mul_terms([ph, "uxxx_0"]), self.mul_terms(["u_0", "ux_0"])]]
        return op_list, term_list

    def generate_kdv(self, rng, train, ICs=None,coeff = None):
        delta = 0.022
        delta_range = self.get_sample_range(delta)

        item = {"type": "kdv"}

        delta = self.refine_floats(rng.uniform(*delta_range, (1,)))[0]
        delta2 = delta**2
        tf = self.tfinals["kdv"]
        p = self.params
        coeff = p.t_range / tf

        op_list = [["add", "add"]]
        term_list = [
            [
                self.mul_terms([str(coeff), "ut_0"]),
                self.mul_terms([str(delta2), "uxxx_0"]),
                self.mul_terms(["u_0", "ux_0"]),
            ]
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        # Assuming nx is even for simplicity
        kx = np.fft.fftfreq(p.x_num, d=p.x_range / p.x_num)
        kx = 2.0 * np.pi * kx  # Scale by 2*pi for spatial frequency

        def uhat2vhat(t, uhat):
            return np.exp(-1j * (kx**3) * delta2 * t) * uhat

        def vhat2uhat(t, vhat):
            return np.exp(1j * (kx**3) * delta2 * t) * vhat

        # ----- Define RHS -----
        def uhatprime(t, uhat):
            u = np.fft.ifft(uhat)
            return 1j * (kx**2) * delta2 * uhat - 0.5j * kx * np.fft.fft(u**2)

        def vhatprime(t, vhat):
            u = np.fft.ifft(vhat2uhat(t, vhat))
            return -0.5j * kx * uhat2vhat(t, np.fft.fft(u**2))

        # u0 = np.cos(np.pi * self.x_grid)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        # if p.IC_types == "train":
        #     y_0s = np.array(
        #         init_multi(self.x_grid.flatten(), numbers=num_initial_points * 10, k_tot=2, num_choise_k=1,
        #                    init_key=rng.randint(100000), if_norm= True))
        # else:
        #     y_0s = np.array(
        #         generate_gaussian_process_jax(self.x_grid.flatten(), init_key=rng.randint(100000),
        #                                       num=num_initial_points * 10, kernel=rbf_kernel_jax,
        #                                       k_sigma=1, k_l=.8))

        if ICs is not None:
            y_0s = np.array(ICs)

        res = []

        for i in range(num_initial_points * 10):
            if ICs is not None:
                u0 = y_0s[i,:]
            elif p.IC_types == "train":
                base_frequency = 2 * np.pi / p.x_range
                n1, n2 = rng.randint(1, 3, size=2)
                frequencies = base_frequency * np.array([n1, n2])

                random_phases = rng.uniform(0, 2 * np.pi, size=2)
                random_amplitudes = rng.uniform(0, 1, size=2)

                # Composite wave function
                def _func(x):
                    # return random_amplitudes[0] * np.sin(
                    #     base_frequency * x + random_phases[0])
                    wave1 = random_amplitudes[0] * np.sin(frequencies[0] * x + random_phases[0])
                    wave2 = random_amplitudes[1] * np.sin(frequencies[1] * x + random_phases[1])
                    return wave1 + wave2

                vec = _func(self.x_grid.flatten())
                slope = vec[-1] - vec[0]
                slope /= p.x_range
                vec = vec - slope * self.x_grid.flatten()
                min, max = np.min(vec), np.max(vec)

                def func(x):
                    val = _func(x)
                    linear = slope * x
                    val = val - linear
                    val = (val - min) / (2 * (max - min))
                    return val

                u0 = func(self.x_grid.flatten())
            else:
                means = rng.uniform(0.5, 1.5, size=2)
                # means2 = rng.uniform(1, 1.5)
                std_devs = rng.uniform(0.3, 0.5, size=2)  # Random standard deviations
                sign = rng.randint(2, size=2) * 2 - 1  # Random standard deviations

                # Define the composite Gaussian function
                def _func(x):
                    gaussian1 = np.exp(-((x - means[0]) ** 2) / (2 * std_devs[0] ** 2))
                    gaussian2 = np.exp(-((x - means[1]) ** 2) / (2 * std_devs[1] ** 2))

                    return sign[0] * gaussian1 + sign[1] * gaussian2

                vec = _func(self.x_grid.flatten())
                slope = vec[-1] - vec[0]
                slope /= p.x_range
                vec = vec - slope * self.x_grid.flatten()
                min, max = np.min(vec), np.max(vec)

                def func(x):
                    val = _func(x)
                    linear = slope * x
                    val = val - linear
                    val = (val - min) / (2 * (max - min))
                    return val

                u0 = func(self.x_grid.flatten())
            uhat0 = np.fft.fft(u0)
            vhat0 = uhat2vhat(0, uhat0)
            try:
                sol = solve_ivp(
                    vhatprime,
                    [t / coeff for t in self.t_span],
                    vhat0,
                    method="RK45",
                    t_eval=self.t_eval / coeff,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    vhat = sol.y
                    # u = np.fft.ifft(vhat2uhat(self.t_eval/coeff, vhat))
                    u = np.zeros((p.x_num, p.t_num), dtype=complex)
                    for i in range(p.t_num):
                        u[:, i] = np.fft.ifft(vhat2uhat(self.t_eval[i] / coeff, vhat[:, i]))
                    if np.all(np.abs(np.imag(u)) < 0.05):
                        u = np.real(u)
                    else:
                        raise ValueError

                    res.append(torch.from_numpy(u.transpose().astype(np.single)).unsqueeze(-1))
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        #
        # for i in range(len(res)):
        #     this_sample = np.asarray(res[i])[:, :, 0]
        #     sns.heatmap(this_sample)
        #     print(np.linalg.norm(this_sample[32:,:]))
        #     print(np.linalg.norm(this_sample[32:, :] - np.mean(this_sample[32:, :])))
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title('kdv Equation with coefficient = ' + str(delta))
        #     plt.show()
        #     plt.close()
        #     for j  in range(this_sample.shape[0]):
        #         plt.plot(this_sample[j,:])
        #     plt.plot(this_sample[0, :],linewidth = 5)
        #     plt.show()
        #     plt.close()


        return item

    #
    # def twobody_diff_react_1D_tree_list(self):
    #     p = self.params
    #     ph = self.ph
    #     op_list = [["sub", "sub"], ["sub", "sub"]]
    #     term_list = [
    #         [
    #             self.mul_terms([ph, "ut_0"]),
    #             self.mul_terms([ph, "uxx_0"]),
    #             self.add_terms([
    #                 Node("u_0", p),
    #                 Node("neg", p, [self.mul_terms(["u_0", "u_0","u_0"])]),
    #                 Node("neg", p, [Node("u_1",p)]),
    #                 Node(ph,p)
    #             ])
    #         ],
    #         [
    #             self.mul_terms([ph, "ut_1"]),
    #             self.mul_terms([ph, "uxx_1"]),
    #             Node("mul",p,[Node("sub",p,[Node("u_0",p),Node("u_1",p)]),Node(ph,p)])
    #         ]
    #     ]
    #     return op_list, term_list
    #
    # def generate_twobody_diff_react_1D(self, rng, train):
    #     Da, Db, alpha, beta = 20, 2000, -0.005, 10
    #
    #     Da_range = self.get_sample_range(Da)
    #     Db_range = self.get_sample_range(Db)
    #     alpha_range = self.get_sample_range(alpha)
    #     beta_range = self.get_sample_range(beta)
    #
    #     item = {"type": "twobody_diff_react_1D"}
    #
    #     Da = self.refine_floats(rng.uniform(*Da_range, (1,)))[0]
    #     Db = self.refine_floats(rng.uniform(*Db_range, (1,)))[0]
    #     alpha = self.refine_floats(rng.uniform(*alpha_range, (1,)))[0]
    #     beta = self.refine_floats(rng.uniform(*beta_range, (1,)))[0]
    #
    #     p = self.params
    #
    #     dx =  p.x_range / p.x_num
    #     adj_coeff = 1
    #     while dx < 0.3:
    #         dx = dx * 10
    #         adj_coeff *= 10
    #     Da /= adj_coeff **2
    #     Db /= adj_coeff **2
    #
    #     dt = 1 / (p.t_num - 1)
    #
    #     adj_coeff_t = p.t_range
    #
    #     t_step = 1
    #     while dt > 0.0002:
    #         dt = dt / 2
    #         t_step *= 2
    #
    #     op_list = [["sub", "sub"], ["sub", "sub"]]
    #     term_list = [
    #         [
    #             self.mul_terms([str(adj_coeff_t), "ut_0"]),
    #             self.mul_terms([str(Da), "uxx_0"]),
    #             self.add_terms([
    #                 Node("u_0", p),
    #                 Node("neg", p, [self.mul_terms(["u_0", "u_0","u_0"])]),
    #                 Node("neg", p, [Node("u_1",p)]),
    #                 Node(str(alpha),p)
    #             ])
    #         ],
    #         [
    #             self.mul_terms([str(adj_coeff_t), "ut_1"]),
    #             self.mul_terms([str(Db), "uxx_1"]),
    #             Node("mul",p,[Node("sub",p,[Node("u_0",p),Node("u_1",p)]),Node(str(beta),p)])
    #         ]
    #     ]
    #
    #     item["tree"] = self.tree_from_list(op_list, term_list)
    #
    #     num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
    #
    #     def Ra(a, b): return a - a ** 3 - b + alpha
    #     def Rb(a, b): return (a - b) * beta
    #
    #     res = []
    #
    #     try:
    #         for i in range(num_initial_points):
    #             RDE = OneDimensionalRDEquations(
    #                 Da, Db, Ra, Rb, rng,
    #                 num_x=p.x_num, dx=dx, dt=dt,
    #                 t_steps=t_step
    #             )
    #
    #             a, b = RDE.getwholeiter(p.t_num-1)
    #             uu = torch.from_numpy(np.stack((a,b),axis=2))
    #             res.append(uu)
    #     except Exception as e:
    #         pass
    #     item["data"] = res
    #
    #
    #     # for i in range(len(res)):
    #     #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with two subplots
    #     #
    #     #     # Plot the first heatmap
    #     #     sns.heatmap(np.asarray(res[i])[:, :, 0], ax=axes[0])
    #     #     axes[0].set_xlabel('x')
    #     #     axes[0].set_ylabel('t')
    #     #
    #     #     # Plot the second heatmap
    #     #     sns.heatmap(np.asarray(res[i])[:, :, 1], ax=axes[1])
    #     #     axes[1].set_xlabel('x')
    #     #     axes[1].set_ylabel('t')
    #     #
    #     #     plt.suptitle('Two body diffusion reaction with diffusion coefficient = ' +str(Da ) + "and" + str(Db))
    #     #
    #     #     plt.tight_layout()  # Adjusts the subplots to fit into the figure area.
    #     #     plt.show()
    #
    #     return item

    def advection_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add"]]
        term_list = [
            [
                self.mul_terms([ph, "ut_0"]),
                self.mul_terms([ph, "ux_0"]),
            ]
        ]
        return op_list, term_list

    def generate_advection(self, rng, train, ICs=None,coeff = None):
        p = self.params
        if p.extrapolate_pdetypes:
            beta = 1
        else:
            beta = 0.5

        tf = self.tfinals["advection"]
        coeff = p.t_range/tf

        beta_range = self.get_sample_range(beta)
        item = {"type": "advection"}

        beta = self.refine_floats(rng.uniform(*beta_range, (1,)))[0]

        op_list = [["add"]]
        term_list = [
            [
                self.mul_terms([str(coeff), "ut_0"]),
                self.mul_terms([str(beta), "ux_0"]),
            ]
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        res = []
        for i in range(num_initial_points * 10):
            if p.IC_types == "train":
                base_frequency = 2 * np.pi / p.x_range
                n1, n2 = rng.randint(1, 3, size=2)
                frequencies = base_frequency * np.array([n1, n2])

                random_phases = rng.uniform(0, 2 * np.pi, size=2)
                random_amplitudes = rng.uniform(0, 1, size=2)

                # Composite wave function
                def _func(x):
                    # return random_amplitudes[0] * np.sin(
                    #     base_frequency * x + random_phases[0])
                    wave1 = random_amplitudes[0] * np.sin(frequencies[0] * x + random_phases[0])
                    wave2 = random_amplitudes[1] * np.sin(frequencies[1] * x + random_phases[1])
                    return wave1 + wave2

                vec = _func(self.x_grid.flatten())
                slope = vec[-1] - vec[0]
                slope /= p.x_range
                vec = vec - slope * self.x_grid.flatten()
                min, max = np.min(vec), np.max(vec)

                def func(x):
                    val = _func(x)
                    linear = slope * x
                    val = val - linear
                    val = (val - min) / (max - min)
                    return val

            else:
                means = rng.uniform(0.5, 1.5, size=2)
                # means2 = rng.uniform(1,1.5)
                std_devs = rng.uniform(0.1, 0.5, size=2)  # Random standard deviations
                sign = rng.randint(2, size=2) * 2 - 1  # Random standard deviations

                # Define the composite Gaussian function
                def _func(x):
                    gaussian1 = np.exp(-((x - means[0]) ** 2) / (2 * std_devs[0] ** 2))
                    gaussian2 = np.exp(-((x - means[1]) ** 2) / (2 * std_devs[1] ** 2))

                    return sign[0] * gaussian1 + sign[1] * gaussian2

                vec = _func(self.x_grid.flatten())
                slope = vec[-1] - vec[0]
                slope /= p.x_range
                vec = vec - slope * self.x_grid.flatten()
                min, max = np.min(vec), np.max(vec)

                def func(x):
                    val = _func(x)
                    linear = slope * x
                    val = val - linear
                    val = (val - min) / (max - min)
                    return val

            try:
                # y0 = func(self.x_grid.flatten())
                # max_y0,min_y0 = np.max(y0),np.min(y0)
                y = [func(self.x_grid.flatten())]
                t_eval = self.t_eval / coeff
                for cur_t in t_eval[1:]:
                    x_adjusted = (self.x_grid.flatten() - beta * cur_t) % p.x_range
                    y.append(func(x_adjusted))
                y = np.array(y)
                res.append(torch.from_numpy(y.astype(np.single)).unsqueeze(-1))
                if len(res) >= num_initial_points:
                    break
            except Exception as e:
                pass
        item["data"] = res
        # CFL = 0.4
        # uu= adv_gen_f(p.x_range,0.,p.x_num,0.,p.t_range, p.t_range / (p.t_num - 1), CFL, 1.0, num_initial_points, 200,
        #               rng.randint(100000),  beta)
        #
        # item["data"] =    [torch.tensor(np.array( jnp.transpose(uu[:,i, :, :], axes = (1,2,0)))) for i in range(uu.shape[1])]

        # uu_np = [np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0))) for i in
        #          range(uu.shape[1])]

        # for i in range(len(res)):
        #     this_sample = np.asarray(res[i])[:, :, 0]
        #     sns.heatmap(this_sample)
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title('Advection Equation with coefficient = ' + str(beta))
        #     plt.show()
        #     plt.close()
        #
        #     plt.plot(this_sample[0,:])
        #     plt.show()
        #     plt.close()

        return item

    def wave_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["sub"]]
        if p.extrapolate_pdetypes:
            term_list = [
                [
                    self.mul_terms([ph, "utt_0"]),
                    self.mul_terms([ph, "uxx_0"]),
                ]
            ]
        else:
            term_list = [
                [
                    Node("utt_0", p),
                    self.mul_terms([ph, "uxx_0"]),
                ]
            ]
        return op_list, term_list

    def generate_wave(self, rng, train, ICs=None,coeff = None):
        #  Assume initial velocity is 0

        item = {"type": "wave"}

        p = self.params

        op_list = [["sub"]]
        if p.extrapolate_pdetypes:
            beta = 1
        else:
            beta = 0.5

        tf = self.tfinals["wave"]
        coeff_t = p.t_range / tf
        t_eval = self.t_eval / coeff_t
        term_list = [
            [
                self.mul_terms([str(coeff_t ** 2), "utt_0"]),
                self.mul_terms([str(beta ** 2), "uxx_0"]),
            ]
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)
        beta_range = self.get_sample_range(beta)
        beta = self.refine_floats(rng.uniform(*beta_range, (1,)))[0]
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        res = []
        for i in range(num_initial_points * 10):
            if p.IC_types == "train":
                base_frequency = 2 * np.pi / p.x_range
                n1, n2 = rng.randint(1, 3, size=2)
                frequencies = base_frequency * np.array([n1, n2])

                random_phases = rng.uniform(0, 2 * np.pi, size=2)
                random_amplitudes = rng.uniform(0, 1, size=2)

                # Composite wave function
                def _func(x):
                    # return random_amplitudes[0] * np.sin(
                    #     base_frequency * x + random_phases[0])
                    wave1 = random_amplitudes[0] * np.sin(frequencies[0] * x + random_phases[0])
                    wave2 = random_amplitudes[1] * np.sin(frequencies[1] * x + random_phases[1])
                    return wave1 + wave2

                vec = _func(self.x_grid.flatten())
                slope = vec[-1] - vec[0]
                slope /= p.x_range
                vec = vec - slope * self.x_grid.flatten()
                min, max = np.min(vec), np.max(vec)

                def func(x):
                    val = _func(x)
                    linear = slope * x
                    val = val - linear
                    val = (val - min) / (max - min)
                    return val

            else:
                means = rng.uniform(0.5, 1.5, size=2)
                # means2 = rng.uniform(1,1.5)
                std_devs = rng.uniform(0.1, 0.5, size=2)  # Random standard deviations
                sign = rng.randint(2, size=2) * 2 - 1  # Random standard deviations

                # Define the composite Gaussian function
                def _func(x):
                    gaussian1 = np.exp(-((x - means[0]) ** 2) / (2 * std_devs[0] ** 2))
                    gaussian2 = np.exp(-((x - means[1]) ** 2) / (2 * std_devs[1] ** 2))

                    return sign[0] * gaussian1 + sign[1] * gaussian2

                vec = _func(self.x_grid.flatten())
                slope = vec[-1] - vec[0]
                slope /= p.x_range
                vec = vec - slope * self.x_grid.flatten()
                min, max = np.min(vec), np.max(vec)

                def func(x):
                    val = _func(x)
                    linear = slope * x
                    val = val - linear
                    val = (val - min) / (max - min)
                    return val

            try:
                y = [func(self.x_grid.flatten())]
                for cur_t in t_eval[1:]:
                    x_adjusted1 = (self.x_grid.flatten() - beta * cur_t) % p.x_range
                    x_adjusted2 = (self.x_grid.flatten() + beta * cur_t) % p.x_range
                    y.append(0.5 * func(x_adjusted1) + 0.5 * func(x_adjusted2))
                y = np.array(y)
                res.append(torch.from_numpy(y.astype(np.single)).unsqueeze(-1))
                if len(res) >= num_initial_points:
                    break
            except Exception as e:
                pass
        item["data"] = res

        # for i in range(len(res)):
        #     this_sample = np.asarray(res[i])[:, :, 0]
        #     sns.heatmap(this_sample)
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title('wave Equation  with coeff = ' + str(beta))
        #     plt.show()
        #     plt.close()
        #     for j in range(p.t_num):
        #         plt.plot(this_sample[j,:])
        #     plt.plot(this_sample[0, :],linewidth = 5)
        #     plt.show()
        #     plt.close()
        return item

    def diff_logisreact_1D_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["sub", "sub"]]
        term_list = [
            [
                self.mul_terms([ph, "ut_0"]),
                self.mul_terms([ph, "uxx_0"]),
                Node(
                    "mul",
                    p,
                    [Node(ph, p), Node("mul", p, [Node("u_0", p), Node("sub", p, [Node(ph, p), Node("u_0", p)])])],
                ),
                # self.mul_terms([ph, "u_0"]),
                # self.mul_terms([ph, "u_0", "u_0"]),
            ]
        ]
        return op_list, term_list

    def generate_diff_logisreact_1D(self, rng, train, ICs=None,coeff = None):
        p = self.params

        if p.extrapolate_pdetypes:
            rho = .1
            nu = .01
        else:
            rho = 1
            nu = 3e-3
        item = {"type": "diff_logisreact_1D"}

        nu_range = self.get_sample_range(nu)
        rho_range = self.get_sample_range(rho)

        nu = self.refine_floats(rng.uniform(*nu_range, (1,)))[0]
        rho = self.refine_floats(rng.uniform(*rho_range, (1,)))[0]

        tf = self.tfinals["diff_logisreact_1D"]
        coeff_t = p.t_range / tf
        op_list = [["sub", "sub"]]
        term_list = [
            [
                self.mul_terms([str(coeff_t), "ut_0"]),
                self.mul_terms([str(nu), "uxx_0"]),
                Node(
                    "mul",
                    p,
                    [
                        Node(str(rho), p),
                        Node("mul", p, [Node("u_0", p), Node("sub", p, [Node(str(1), p), Node("u_0", p)])]),
                    ],
                ),
                # self.mul_terms([str(rho), "u_0"]),
                # self.mul_terms([str(rho), "u_0", "u_0"]),
            ]
        ]

        item["tree"] = self.tree_from_list(op_list, term_list)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        try:
            if ICs is not None:
                GivenIC = ICs
                IC_train = True
                numbers = jnp.shape(ICs)[0]
            elif p.IC_types == "train":
                IC_train = True
                GivenIC = None
                numbers = num_initial_points * 10
            else:
                IC_train = False
                GivenIC = None
                numbers = num_initial_points * 10
            CFL = 0.35
            uu = diff_react_1D_f(
                p.x_range,
                0.0,
                p.x_num,
                0.0,
                p.t_range/coeff_t,
                self.dt/coeff_t,
                p.t_num,
                CFL,
                numbers,
                20,
                rng.randint(100000),
                rho,
                nu,
                IC_train=IC_train,
                GivenIC = GivenIC,
            )
            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample))
                    res_np.append(sample)

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        # item["data"] =    [torch.tensor(np.array( jnp.transpose(uu[:,i, :, :], axes = (1,2,0)))) for i in range(uu.shape[1])]

        # uu_np = [np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0))) for i in
        #          range(uu.shape[1])]
        item["data"] = res
        # for i in range(len(res)):
        #     sns.heatmap(res_np[i][:, :, 0])
        #     print(np.linalg.norm(res_np[i][32:,:,:])/(np.linalg.norm(res_np[i][32:, :, :] - np.mean(res_np[i][32:, :, :]))))
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title('Diffusion Reaction Equation with coefficient = ' + str(nu) + "and" + str(rho))
        #     plt.show()
        #
        #     this = res_np[i]
        #     plt.plot(this[0,:])
        #     plt.show()
        #     plt.close()
        #
        return item

    def diff_linearreact_1D_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["sub", "sub"]]
        term_list = [
            [
                self.mul_terms([ph, "ut_0"]),
                self.mul_terms([ph, "uxx_0"]),
                self.mul_terms([ph, "u_0"]),
            ]
        ]
        return op_list, term_list

    def generate_diff_linearreact_1D(self, rng, train, ICs=None,coeff = None):
        p = self.params
        if p.extrapolate_pdetypes:
            rho = .1
            nu = .01
        else:
            rho = .1
            nu = 3e-3
        item = {"type": "diff_linearreact_1D"}

        nu_range = self.get_sample_range(nu)
        rho_range = self.get_sample_range(rho)

        nu = self.refine_floats(rng.uniform(*nu_range, (1,)))[0]
        rho = self.refine_floats(rng.uniform(*rho_range, (1,)))[0]

        tf = self.tfinals["diff_linearreact_1D"]
        coeff_t = p.t_range/tf

        op_list = [["sub", "sub"]]
        term_list = [
            [
                self.mul_terms([str(coeff_t), "ut_0"]),
                self.mul_terms([str(nu), "uxx_0"]),
                self.mul_terms([str(rho), "u_0"]),
            ]
        ]

        item["tree"] = self.tree_from_list(op_list, term_list)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        try:
            if ICs is not None:
                GivenIC = ICs
                IC_train = True
                numbers = jnp.shape(ICs)[0]
            elif p.IC_types == "train":
                IC_train = True
                GivenIC = None
                numbers = num_initial_points * 10
            else:
                IC_train = False
                GivenIC = None
                numbers = num_initial_points * 10
            CFL = 0.35
            uu = diff_react_1D_f(
                p.x_range,
                0.0,
                p.x_num,
                0.0,
                p.t_range/coeff_t,
                self.dt/coeff_t,
                p.t_num,
                CFL,
                numbers,
                20,
                rng.randint(100000),
                rho,
                nu,
                react_term="linear",
                IC_train=IC_train,
                GivenIC = GivenIC,
            )
            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample))
                    res_np.append(sample)

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass
        item["data"] = res
        # item["data"] =    [torch.tensor(np.array( jnp.transpose(uu[:,i, :, :], axes = (1,2,0)))) for i in range(uu.shape[1])]

        # uu_np = [np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0))) for i in
        #          range(uu.shape[1])]
        # item["data"] = res
        # for i in range(len(res)):
        #     sns.heatmap(res_np[i][:, :, 0])
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title(
        #         'Diffusion Reaction Equation with coefficient = ' + str(nu) + "and" + str(rho))
        #     plt.show()
        #
        #     this = res_np[i]
        #     plt.plot(this[0,:])
        #     plt.show()
        #     plt.close()
        return item

    def diff_bistablereact_1D_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["sub", "sub"]]
        term_list = [
            [
                self.mul_terms([ph, "ut_0"]),
                self.mul_terms([ph, "uxx_0"]),
                Node(
                    "mul",
                    p,
                    [
                        Node(
                            "mul",
                            p,
                            [
                                Node(ph, p),
                                Node("mul", p, [Node("u_0", p), Node("sub", p, [Node(str(1), p), Node("u_0", p)])]),
                            ],
                        ),
                        Node("sub", p, [Node("u_0", p), Node(ph, p)]),
                    ],
                ),
            ]
        ]
        return op_list, term_list

    def generate_diff_bistablereact_1D(self, rng, train, ICs=None,coeff = None):
        p = self.params
        if p.extrapolate_pdetypes:
            rho = .1
            nu = .01
        else:
            rho = 1
            nu = 3e-3
        item = {"type": "diff_bistablereact_1D"}

        nu_range = self.get_sample_range(nu)
        rho_range = self.get_sample_range(rho)

        nu = self.refine_floats(rng.uniform(*nu_range, (1,)))[0]
        rho = self.refine_floats(rng.uniform(*rho_range, (1,)))[0]
        a = 0


        tf = self.tfinals["diff_bistablereact_1D"]
        coeff = tf/p.t_range

        op_list = [["sub", "sub"]]
        term_list = [
            [
                self.mul_terms([str(coeff), "ut_0"]),
                self.mul_terms([str(nu), "uxx_0"]),
                Node(
                    "mul",
                    p,
                    [
                        Node(
                            "mul",
                            p,
                            [
                                Node(str(rho), p),
                                Node("mul", p, [Node("u_0", p), Node("sub", p, [Node(str(1), p), Node("u_0", p)])]),
                            ],
                        ),
                        Node("sub", p, [Node("u_0", p), Node(str(a), p)]),
                    ],
                ),
            ]
        ]

        item["tree"] = self.tree_from_list(op_list, term_list)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        try:
            if ICs is not None:
                GivenIC = ICs
                IC_train = True
                numbers = jnp.shape(ICs)[0]
            elif p.IC_types == "train":
                IC_train = True
                GivenIC = None
                numbers = num_initial_points * 10
            else:
                IC_train = False
                GivenIC = None
                numbers = num_initial_points * 10
            CFL = 0.35
            uu = diff_react_1D_f(
                p.x_range,
                0.0,
                p.x_num,
                0.0,
                p.t_range/coeff,
                self.dt/coeff,
                p.t_num,
                CFL,
                numbers,
                20,
                rng.randint(100000),
                rho,
                nu,
                react_term="bistable",
                IC_train=IC_train,
                GivenIC = GivenIC
            )
            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample))
                    res_np.append(sample)

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass
        item["data"] = res
        # item["data"] =    [torch.tensor(np.array( jnp.transpose(uu[:,i, :, :], axes = (1,2,0)))) for i in range(uu.shape[1])]

        # uu_np = [np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0))) for i in
        #          range(uu.shape[1])]
        # item["data"] = res
        # for i in range(len(res)):
        #     sns.heatmap(res_np[i][:, :, 0])
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title(
        #         'Diffusion Reaction Equation with coefficient = ' + str(nu) + "and" + str(rho))
        #     plt.show()

        return item

    def diff_squarelogisticreact_1D_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["sub", "sub"]]
        term_list = [
            [
                self.mul_terms([ph, "ut_0"]),
                self.mul_terms([ph, "uxx_0"]),
                Node(
                    "mul",
                    p,
                    [
                        Node(ph, p),
                        Node(
                            "mul",
                            p,
                            [
                                Node("pow2", p, [Node("u_0", p)]),
                                Node("pow2", p, [Node("sub", p, [Node(str(1), p), Node("u_0", p)])]),
                            ],
                        ),
                    ],
                ),
            ]
        ]
        return op_list, term_list

    def generate_diff_squarelogisticreact_1D(self, rng, train, ICs=None,coeff = None):
        p = self.params
        if p.extrapolate_pdetypes:
            rho = .1
            nu = .01
        else:
            rho = 1
            nu = 3e-3
        item = {"type": "diff_squarelogisticreact_1D"}

        nu_range = self.get_sample_range(nu)
        rho_range = self.get_sample_range(rho)

        nu = self.refine_floats(rng.uniform(*nu_range, (1,)))[0]
        rho = self.refine_floats(rng.uniform(*rho_range, (1,)))[0]

        tf = self.tfinals["diff_squarelogisticreact_1D"]
        coeff = p.t_range /tf

        op_list = [["sub", "sub"]]
        term_list = [
            [
                self.mul_terms([str(tf), "ut_0"]),
                self.mul_terms([str(nu), "uxx_0"]),
                Node(
                    "mul",
                    p,
                    [
                        Node(str(rho), p),
                        Node(
                            "mul",
                            p,
                            [
                                Node("pow2", p, [Node("u_0", p)]),
                                Node("pow2", p, [Node("sub", p, [Node(str(1), p), Node("u_0", p)])]),
                            ],
                        ),
                    ],
                ),
            ]
        ]

        item["tree"] = self.tree_from_list(op_list, term_list)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        try:
            if ICs is not None:
                GivenIC = ICs
                IC_train = True
                numbers = jnp.shape(ICs)[0]
            elif p.IC_types == "train":
                IC_train = True
                GivenIC = None
                numbers = num_initial_points * 10
            else:
                IC_train = False
                GivenIC = None
                numbers = num_initial_points * 10
            CFL = 0.35
            uu = diff_react_1D_f(
                p.x_range,
                0.0,
                p.x_num,
                0.0,
                p.t_range/coeff,
                self.dt/coeff,
                p.t_num,
                CFL,
                numbers,
                20,
                rng.randint(100000),
                rho,
                nu,
                react_term="squarelogistic",
                IC_train=IC_train,
                GivenIC = GivenIC,
            )
            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample))
                    res_np.append(sample)

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res

        return item

    def burgers_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add", "sub"]]
        term_list = [
            [
                self.mul_terms([ph, "ut_0"]),
                self.mul_terms([ph, "u_0", "ux_0"]),
                self.mul_terms([ph, "uxx_0"]),
            ]
        ]
        return op_list, term_list

    def generate_burgers(self, rng, train, ICs=None,coeff = None):

        item = {"type": "burgers"}
        p = self.params
        if coeff is not None:
            eps = coeff[0] * np.pi
            k = coeff[1]
        else:
            if p.extrapolate_pdetypes:
                eps = .03
            else:
                eps = 0.01  # .05
            k = 1

            eps_range = self.get_sample_range(eps)
            k_range = self.get_sample_range(k)

            eps = self.refine_floats(rng.uniform(*eps_range, (1,)))[0]
            k = self.refine_floats(rng.uniform(*k_range, (1,)))[0]

        tf = self.tfinals["burgers"]
        coeff = p.t_range/tf

        op_list = [["add", "sub"]]
        term_list = [
            [
                self.mul_terms([str(coeff), "ut_0"]),
                self.mul_terms([str(k), "u_0", "ux_0"]),
                self.mul_terms([str(eps / np.pi), "uxx_0"]),
            ]
        ]

        item["tree"] = self.tree_from_list(op_list, term_list)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        CFL = 0.4
        train_num = 0
        train_num2 = 0
        if p.IC_types.startswith("rarefactiontrain"):
            _, train_num = p.IC_types.split("_")
            train_num = int(train_num)
        elif p.IC_types.startswith("rarefaction2train"):
            _, train_num2 = p.IC_types.split("_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            numbers = jnp.shape(ICs)[0]
            mode = "periodic"
        elif p.IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif p.IC_types == "one_shock" or p.IC_types == "one_and_two_shock" or  train_num > 0 or train_num2 > 0:
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                            self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "rarefaction":
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "two_shocks":
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end,(numbers, 3))
            ends = np.sort(ends, axis=1)
            breakmid1 = rng.uniform(p.x_range * 0.2, p.x_range * 0.5, (numbers,))
            breakmid2 = rng.uniform(p.x_range * 0.5, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 2] * (self.x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                            self.x_grid.flatten() < breakmid2[i]) * (
                                     self.x_grid.flatten() >= breakmid1[i]) + ends[i, 0] * (
                                     self.x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        uu = burgers_f(
            p.x_range,
            0.0,
            p.x_num,
            0.0,
            p.t_range / coeff,
            self.dt / coeff,
            p.t_num,
            CFL,
            numbers,
            20,
            rng.randint(100000),
            eps,
            k,
            fluxx="quadratic",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )

        try:

            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample))
                    res_np.append(sample)

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res

        # uu_np = [np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0))) for i in
        #          range(uu.shape[1])]

        # for i in range(len(res)):
        #     sns.heatmap(res_np[i][:, :, 0])
        #     # print(np.linalg.norm(res_np[i][32:,:,:]))
        #     print(np.linalg.norm(res_np[i][32:,:,:])/(np.linalg.norm(res_np[i][32:, :, :] - np.mean(res_np[i][32:, :, :]))))
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title(
        #         'Burgers Equation with coefficient = ' + str(eps) + "and" + str(k))
        #     plt.show()
        #     for j in range(p.t_num):
        #         plt.plot(res_np[i][j,:])
        #     plt.plot(res_np[i][0, :],linewidth = 5)
        #     plt.show()

        return item

    def inviscid_burgers_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add"]]
        term_list = [[self.mul_terms([ph,"ut_0"]), self.mul_terms([ph, "u_0", "ux_0"])]]
        return op_list, term_list

    def generate_inviscid_burgers(self, rng, train, ICs=None, coeff = None):
        item = {"type": "inviscid_burgers"}
        p = self.params
        if coeff is not None:
            k = coeff[1]
        else:
            k = 1
            k_range = self.get_sample_range(k)
            k = self.refine_floats(rng.uniform(*k_range, (1,)))[0]

        tf = self.tfinals["inviscid_burgers"]
        coeff = p.t_range/tf
        op_list = [["add"]]
        term_list = [
            [
                self.mul_terms([str(coeff),"ut_0"]),
                self.mul_terms([str(k), "u_0", "ux_0"]),
            ]
        ]

        item["tree"] = self.tree_from_list(op_list, term_list)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        CFL = 0.4
        train_num = 0
        train_num2 = 0
        if p.IC_types.startswith("rarefactiontrain"):
            _,train_num = p.IC_types.split("_")
            train_num = int(train_num)
        elif p.IC_types.startswith("rarefaction2train"):
            _,train_num2 = p.IC_types.split("_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            numbers = jnp.shape(ICs)[0]
            mode = "periodic"
        elif p.IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif p.IC_types == "one_shock" or train_num == 6 or train_num2 >1:
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "rarefaction"  or (train_num >0 and train_num < 6) or train_num2 == 1:
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "two_shocks" or p.IC_types == "one_and_two_shock":
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 3))
            ends = np.sort(ends, axis=1)
            breakmid1 = rng.uniform(p.x_range * 0.2, p.x_range * 0.5, (numbers,))
            breakmid2 = rng.uniform(p.x_range * 0.5, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 2] * (self.x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        self.x_grid.flatten() < breakmid2[i]) * (
                                 self.x_grid.flatten() >= breakmid1[i]) + ends[i, 0] * (
                                 self.x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        uu = burgers_f(
            p.x_range,
            0.0,
            p.x_num,
            0.0,
            p.t_range / coeff,
            self.dt / coeff,
            p.t_num,
            CFL,
            numbers,
            20,
            rng.randint(100000),
            0,
            k,
            viscous=False,
            fluxx="quadratic",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )

        try:
            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample))
                    res_np.append(sample)

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res

        # uu_np = [np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0))) for i in
        #          range(uu.shape[1])]

        # for i in range(len(res)):
        #     sns.heatmap(res_np[i][:, :, 0])
        #     print(np.linalg.norm(res_np[i][32:,:,:])/(np.linalg.norm(res_np[i][32:, :, :] - np.mean(res_np[i][32:, :, :]))))
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title(
        #         'Burgers Equation with coefficient =  '+ str(k))
        #     plt.show()
        #     for j in range(p.t_num):
        #         plt.plot(res_np[i][j,:])
        #     plt.plot(res_np[i][0, :],linewidth = 5)
        #     plt.show()

        return item

    def conservation_linearflux_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add", "sub"]]
        term_list = [
            [
                self.mul_terms([ph,"ut_0"]),
                self.mul_terms([ph, "ux_0"]),
                self.mul_terms([ph, "uxx_0"]),
            ]
        ]
        return op_list, term_list

    def generate_conservation_linearflux(self, rng, train, ICs=None,coeff=None):
        p = self.params
        if coeff is not None:
            eps = coeff[0] * np.pi
            k = coeff[1]
        else:
            if p.extrapolate_pdetypes:
                eps = .03
            else:
                eps = 0.01  # .05
            k = 1

            eps_range = self.get_sample_range(eps)
            k_range = self.get_sample_range(k)

            eps = self.refine_floats(rng.uniform(*eps_range, (1,)))[0]
            k = self.refine_floats(rng.uniform(*k_range, (1,)))[0]

        item = {"type": "conservation_linearflux"}

        tf = self.tfinals["conservation_linearflux"]
        coeff = p.t_range/tf

        op_list = [["add", "sub"]]
        term_list = [
            [
                self.mul_terms([str(coeff),"ut_0"]),
                self.mul_terms([str(k), "ux_0"]),
                self.mul_terms([str(eps / np.pi), "uxx_0"]),
            ]
        ]

        item["tree"] = self.tree_from_list(op_list, term_list)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        CFL = 0.4
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            numbers = jnp.shape(ICs)[0]
            mode = "periodic"
        elif p.IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif p.IC_types == "rarefaction":
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "one_shock":
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "two_shocks":
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 3))
            ends = np.sort(ends, axis=1)
            breakmid1 = rng.uniform(p.x_range * 0.2, p.x_range * 0.5, (numbers,))
            breakmid2 = rng.uniform(p.x_range * 0.5, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 2] * (self.x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        self.x_grid.flatten() < breakmid2[i]) * (
                                 self.x_grid.flatten() >= breakmid1[i]) + ends[i, 0] * (
                                 self.x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        uu = burgers_f(
            p.x_range,
            0.0,
            p.x_num,
            0.0,
            p.t_range / coeff,
            self.dt / coeff,
            p.t_num,
            CFL,
            numbers,
            20,
            rng.randint(100000),
            eps,
            k,
            fluxx="linear",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )
        try:

            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample))
                    res_np.append(sample)

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res
        #
        # for i in range(len(res)):
        #     sns.heatmap(res_np[i][:, :, 0])
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title(
        #         'LINEAR FLUX Equation with coefficient = ' + str(eps) + "and" + str(k))
        #     plt.show()
        #
        #     this_sample = res_np[i]
        #     # print(np.linalg.norm(res_np[i][32:, :, :]) / (
        #     #     np.linalg.norm(res_np[i][32:, :, :] - np.mean(res_np[i][32:, :, :]))))
        #     # for j in range(p.t_num):
        #     plt.plot(this_sample[0,:])
        #     plt.show()
        #     plt.close()

        return item

    # def inviscid_conservation_linearflux_tree_list(self):
    #     p = self.params
    #     ph = self.ph
    #     op_list = [["add"]]
    #     term_list = [
    #         [
    #             Node("ut_0", p),
    #             Node("ux_0",p)
    #         ]
    #     ]
    #     return op_list, term_list
    #
    # def generate_inviscid_conservation_linearflux(self, rng, train):
    #     item = {"type": "inviscid_conservation_linearflux"}
    #
    #
    #     p = self.params
    #
    #     op_list = [["add"]]
    #     term_list = [
    #         [
    #             Node("ut_0", p),
    #             Node("ux_0",p),
    #         ]
    #     ]
    #
    #     item["tree"] = self.tree_from_list(op_list, term_list)
    #
    #     num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
    #
    #     CFL = 0.4
    #     uu = burgers_f(p.x_range, 0., p.x_num, 0., p.t_range,self.dt, p.t_num,CFL, num_initial_points,
    #                         20, rng.randint(100000),  0,viscous= False, fluxx = "linear")
    #
    #     item["data"] =    [torch.tensor(np.array( jnp.transpose(uu[:,i, :, :], axes = (1,2,0)))) for i in range(uu.shape[1])]
    #
    #     # uu_np = [np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0))) for i in
    #     #          range(uu.shape[1])]
    #     #
    #     # for i in range(uu.shape[1]):
    #     #     sns.heatmap(uu_np[i][:, :, 0])
    #     #     plt.xlabel('x')
    #     #     plt.ylabel('t')
    #     #     plt.title('Inviscid Linear')
    #     #     plt.show()
    #     #
    #     #     this_sample = uu_np[i]
    #     #
    #     #     plt.plot(this_sample[0,:])
    #     #     plt.show()
    #     #     plt.close()
    #
    #     return item

    def conservation_cubicflux_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add", "sub"]]
        term_list = [
            [
                self.mul_terms([ph,"ut_0"]),
                self.mul_terms([ph, "u_0", "u_0", "ux_0"]),
                self.mul_terms([ph, "uxx_0"]),
            ]
        ]
        return op_list, term_list

    def generate_conservation_cubicflux(self, rng, train, ICs=None,coeff=None):
        p = self.params
        if coeff is not None:
            eps = coeff[0] * np.pi
            k = coeff[1]
        else:
            if p.extrapolate_pdetypes:
                eps = .03
            else:
                eps = 0.01  # .05
            k = 1

            eps_range = self.get_sample_range(eps)
            k_range = self.get_sample_range(k)

            eps = self.refine_floats(rng.uniform(*eps_range, (1,)))[0]
            k = self.refine_floats(rng.uniform(*k_range, (1,)))[0]

        item = {"type": "conservation_cubicflux"}


        tf = self.tfinals["conservation_cubicflux"]
        coeff = p.t_range/tf

        op_list = [["add", "sub"]]
        term_list = [
            [
                self.mul_terms([str(coeff),"ut_0"]),
                self.mul_terms([str(k), "u_0", "u_0", "ux_0"]),
                self.mul_terms([str(eps / np.pi), "uxx_0"]),
            ]
        ]

        item["tree"] = self.tree_from_list(op_list, term_list)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        CFL = 0.4
        train_num = 0
        train_num2 = 0
        if p.IC_types.startswith("rarefactiontrain"):
            _, train_num = p.IC_types.split("rarefactiontrain_")
            train_num = int(train_num)
        elif p.IC_types.startswith("rarefaction2train"):
            _, train_num2 = p.IC_types.split("rarefaction2train_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            numbers = jnp.shape(ICs)[0]
            mode = "periodic"
        elif p.IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif p.IC_types == "rarefaction" or train_num == 1 or (train_num2 <= 2 and train_num2 > 0):
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.sqrt(ends)
            # sqr = ends **2
            # sort_indices = np.argsort(sqr, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "one_shock" or p.IC_types == "one_and_two_shock" or train_num >= 2 or train_num2>2:
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end,(numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.sqrt(ends)
            # sqr = ends ** 2
            # sort_indices = np.argsort(sqr, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "two_shocks":
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 3))
            ends = np.sort(ends, axis=1)
            # ends = np.sqrt(ends)
            # sqr = ends ** 2
            # sort_indices = np.argsort(sqr, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid1 = rng.uniform(p.x_range * 0.2, p.x_range * 0.5, (numbers,))
            breakmid2 = rng.uniform(p.x_range * 0.5, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 2] * (self.x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        self.x_grid.flatten() < breakmid2[i]) * (
                                 self.x_grid.flatten() >= breakmid1[i]) + ends[i, 0] * (
                                 self.x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        # print(rng.randint(100000), eps, k)
        uu = burgers_f(
            p.x_range,
            0.0,
            p.x_num,
            0.0,
            p.t_range / coeff,
            self.dt / coeff,
            p.t_num,
            CFL,
            numbers,
            20,
            rng.randint(100000),
            eps,
            k,
            fluxx="cubic",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )
        try:

            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample))
                    res_np.append(sample)

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res

        # for i in range(len(res)):
        #     sns.heatmap(res_np[i][:, :, 0])
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title(
        #         'Cubic FLUX Equation with coefficient = ' + str(eps) + "and" + str(k))
        #     print(np.linalg.norm(res_np[i][32:, :, :]) / (
        #         np.linalg.norm(res_np[i][32:, :, :] - np.mean(res_np[i][32:, :, :]))))
        #
        #     plt.show()
        #
        #     this_sample = res_np[i]
        #
        #     # for j in range(p.t_num):
        #     plt.plot(this_sample[0, :])
        #     plt.show()
        #     plt.close()

        return item

    def inviscid_conservation_cubicflux_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add"]]
        term_list = [
            [
                self.mul_terms([ph,"ut_0"]),
                self.mul_terms([ph, "u_0", "u_0", "ux_0"]),
            ]
        ]
        return op_list, term_list

    def generate_inviscid_conservation_cubicflux(self, rng, train, ICs=None,coeff=None):

        item = {"type": "inviscid_conservation_cubicflux"}
        p = self.params
        if coeff is not None:
            k = coeff[1]
        else:
            k = 1
            k_range = self.get_sample_range(k)
            k = self.refine_floats(rng.uniform(*k_range, (1,)))[0]

        tf = self.tfinals["inviscid_conservation_cubicflux"]
        coeff = p.t_range/tf
        op_list = [["add"]]
        term_list = [
            [
                self.mul_terms([str(coeff),"ut_0"]),
                self.mul_terms([str(k), "u_0", "u_0", "ux_0"]),
            ]
        ]

        item["tree"] = self.tree_from_list(op_list, term_list)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        CFL = 0.4
        train_num = 0
        train_num2 = 0
        if p.IC_types.startswith("rarefactiontrain"):
            _, train_num = p.IC_types.split("rarefactiontrain_")
            train_num = int(train_num)
        elif p.IC_types.startswith("rarefaction2train"):
            _, train_num2 = p.IC_types.split("rarefaction2train_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            numbers = jnp.shape(ICs)[0]
            mode = "periodic"
        elif p.IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif p.IC_types == "rarefaction" or ( train_num <5 and train_num >0) or (train_num2 <=3 and train_num2 > 0):
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end,(numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.sqrt(ends)
            # sqr = ends ** 2
            # sort_indices = np.argsort(sqr, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "one_shock" or p.IC_types == "one_and_two_shock" or train_num >= 5 or train_num2 > 3:
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.sqrt(ends)
            # sqr = ends ** 2
            # sort_indices = np.argsort(sqr, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "two_shocks":
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end,(numbers, 3))
            ends = np.sort(ends, axis=1)
            # ends = np.sqrt(ends)
            # sqr = ends ** 2
            # sort_indices = np.argsort(sqr, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid1 = rng.uniform(p.x_range * 0.2, p.x_range * 0.5, (numbers,))
            breakmid2 = rng.uniform(p.x_range * 0.5, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 2] * (self.x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        self.x_grid.flatten() < breakmid2[i]) * (
                                 self.x_grid.flatten() >= breakmid1[i]) + ends[i, 0] * (
                                 self.x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        # print(rng.randint(100000), k)
        uu = burgers_f(
            p.x_range,
            0.0,
            p.x_num,
            0.0,
            p.t_range / coeff,
            self.dt / coeff,
            p.t_num,
            CFL,
            numbers,
            20,
            rng.randint(100000),
            0,
            k,
            viscous=False,
            fluxx="cubic",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )
        try:

            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample))
                    res_np.append(sample)

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res

        # for i in range(len(res)):
        #     sns.heatmap(res_np[i][:, :, 0])
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title(
        #         'Cubic FLUX Equation with coefficient = ' + "and" + str(k))
        #     plt.show()
        #
        #     this_sample = res_np[i]
        #
        #     # for j in range(p.t_num):
        #     plt.plot(this_sample[0, :])
        #     plt.show()
        #     plt.close()

        return item

    def conservation_sinflux_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add", "sub"]]
        term_list = [
            [
                self.mul_terms([ph, "ut_0"]),
                Node("mul", p, [Node(ph, p), Node("mul", p, [Node("cos", p, [Node("u_0", p)]), Node("ux_0", p)])]),
                self.mul_terms([ph, "uxx_0"]),
            ]
        ]
        return op_list, term_list

    def generate_conservation_sinflux(self, rng, train, ICs=None,coeff=None):
        p = self.params
        if coeff is not None:
            eps = coeff[0] * np.pi
            k = coeff[1]
        else:
            if p.extrapolate_pdetypes:
                eps = .03
            else:
                eps = 0.01  # .05
            k = 1

            eps_range = self.get_sample_range(eps)
            k_range = self.get_sample_range(k)

            eps = self.refine_floats(rng.uniform(*eps_range, (1,)))[0]
            k = self.refine_floats(rng.uniform(*k_range, (1,)))[0]

        item = {"type": "conservation_sinflux"}


        tf = self.tfinals["conservation_sinflux"]
        coeff = p.t_range/tf


        op_list = [["add", "sub"]]
        term_list = [
            [
                self.mul_terms([str(coeff), "ut_0"]),
                Node("mul", p, [Node(str(k), p), Node("mul", p, [Node("cos", p, [Node("u_0", p)]), Node("ux_0", p)])]),
                self.mul_terms([str(eps / np.pi), "uxx_0"]),
            ]
        ]

        item["tree"] = self.tree_from_list(op_list, term_list)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        CFL = 0.4
        train_num =0
        train_num2 =0
        if p.IC_types.startswith("rarefactiontrain"):
            _, train_num = p.IC_types.split("rarefactiontrain_")
            train_num = int(train_num)
        elif p.IC_types.startswith("rarefaction2train"):
            _, train_num2 = p.IC_types.split("rarefaction2train_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            mode = "periodic"
            numbers = jnp.shape(ICs)[0]
        elif p.IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif p.IC_types == "one_shock" or train_num >= 3 or train_num2 > 4:
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.arccos(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "rarefaction" or (train_num < 3 and train_num > 0) or (
                    train_num2 <= 4 and train_num2 > 0):
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = -np.arccos(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "two_shocks":
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start, p.IC_jump_end, (numbers, 3))
            ends = np.sort(ends, axis=1)
            # ends = np.arccos(ends)
            breakmid1 = rng.uniform(p.x_range * 0.2, p.x_range * 0.5, (numbers,))
            breakmid2 = rng.uniform(p.x_range * 0.5, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (self.x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        self.x_grid.flatten() < breakmid2[i]) * (
                                 self.x_grid.flatten() >= breakmid1[i]) + ends[i, 2] * (
                                 self.x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        uu = burgers_f(
            p.x_range,
            0.0,
            p.x_num,
            0.0,
            p.t_range / coeff,
            self.dt / coeff,
            p.t_num,
            CFL,
            numbers,
            20,
            rng.randint(100000),
            eps,
            k,
            fluxx="sin",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )
        try:
            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 30:
                    res.append(torch.tensor(sample))
                    res_np.append(sample)

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res

        # for i in range(len(res)):
        #     sns.heatmap(res_np[i][:, :, 0])
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title(
        #         'Sin FLUX Equation with coefficient = '+ str(eps)+ "and" + str(k))
        #     plt.show()
        #
        #     this_sample = res_np[i]
        #     print(np.linalg.norm(res_np[i][32:, :, :]) / (
        #            np.linalg.norm(res_np[i][32:, :, :] - np.mean(res_np[i][32:, :, :]))))
        #     # for j in range(p.t_num):
        #     plt.plot(this_sample[0, :])
        #     plt.show()
        #     plt.close()

        return item

    def inviscid_conservation_sinflux_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add"]]
        term_list = [
            [
                self.mul_terms([ph,"ut_0"]),
                Node("mul", p, [Node(ph, p), Node("mul", p, [Node("cos", p, [Node("u_0", p)]), Node("ux_0", p)])]),
            ]
        ]
        return op_list, term_list

    def generate_inviscid_conservation_sinflux(self, rng, train, ICs=None,coeff=None):

        p = self.params
        item = {"type": "inviscid_conservation_sinflux"}
        if coeff is not None:
            k = coeff[1]
        else:
            k = 1

            k_range = self.get_sample_range(k)
            k = self.refine_floats(rng.uniform(*k_range, (1,)))[0]



        tf = self.tfinals["inviscid_conservation_sinflux"]
        coeff = p.t_range/tf

        op_list = [["add"]]
        term_list = [
            [
                self.mul_terms([str(coeff), "ut_0"]),
                Node("mul", p, [Node(str(k), p), Node("mul", p, [Node("cos", p, [Node("u_0", p)]), Node("ux_0", p)])]),
            ]
        ]

        item["tree"] = self.tree_from_list(op_list, term_list)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        CFL = 0.4
        train_num=0
        train_num2 =0
        if p.IC_types.startswith("rarefactiontrain"):
            _, train_num = p.IC_types.split("rarefactiontrain_")
            train_num = int(train_num)
        elif p.IC_types.startswith("rarefaction2train"):
            _, train_num2 = p.IC_types.split("rarefaction2train_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            numbers = jnp.shape(ICs)[0]
            mode = "periodic"
        elif p.IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif p.IC_types == "one_shock" or train_num >= 4 or train_num2 > 5:
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.arccos(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "rarefaction" or (train_num < 4 and train_num > 0) or (
                    train_num2 <= 5 and train_num2 > 0):
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end,(numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.arccos(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "two_shocks":
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end,(numbers, 3))
            ends = np.sort(ends, axis=1)
            # ends = -np.arccos(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid1 = rng.uniform(p.x_range * 0.2, p.x_range * 0.5, (numbers,))
            breakmid2 = rng.uniform(p.x_range * 0.5, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (self.x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        self.x_grid.flatten() < breakmid2[i]) * (
                                 self.x_grid.flatten() >= breakmid1[i]) + ends[i, 2] * (
                                 self.x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        uu = burgers_f(
            p.x_range,
            0.0,
            p.x_num,
            0.0,
            p.t_range / coeff,
            self.dt / coeff,
            p.t_num,
            CFL,
            numbers,
            20,
            rng.randint(100000),
            0,
            k,
            viscous=False,
            fluxx="sin",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )
        try:

            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 30:
                    res.append(torch.tensor(sample))
                    res_np.append(sample)

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res

        # for i in range(len(res)):
        #     sns.heatmap(res_np[i][:, :, 0])
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title(
        #         'Sin FLUX Equation with coefficient = ' + "and" + str(k))
        #     plt.show()
        #
        #     this_sample = res_np[i]
        #     print(np.linalg.norm(res_np[i][32:, :, :]) / (
        #         np.linalg.norm(res_np[i][32:, :, :] - np.mean(res_np[i][32:, :, :]))))
        #
        #     # for j in range(p.t_num):
        #     plt.plot(this_sample[0, :])
        #     plt.show()
        #     plt.close()
        #
        return item


    def conservation_cosflux_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["sub", "sub"]]
        term_list = [
            [
                self.mul_terms([ph, "ut_0"]),
                Node("mul", p, [Node(ph, p), Node("mul", p, [Node("sin", p, [Node("u_0", p)]), Node("ux_0", p)])]),
                self.mul_terms([ph, "uxx_0"]),
            ]
        ]
        return op_list, term_list

    def generate_conservation_cosflux(self, rng, train, ICs=None,coeff=None):
        p = self.params
        if coeff is not None:
            eps = coeff[0] * np.pi
            k = coeff[1]
        else:
            if p.extrapolate_pdetypes:
                eps = .03
            else:
                eps = 0.01  # .05
            if p.IC_types.startswith("rarefaction") or p.IC_types == "one_shock" or p.IC_types == "two_shocks":
                k=-1
            else:
                k =1

            eps_range = self.get_sample_range(eps)
            k_range = self.get_sample_range(k)

            eps = self.refine_floats(rng.uniform(*eps_range, (1,)))[0]
            k = self.refine_floats(rng.uniform(*k_range, (1,)))[0]

        item = {"type": "conservation_cosflux"}


        tf = self.tfinals["conservation_cosflux"]
        coeff = p.t_range/tf


        op_list = [["sub", "sub"]]
        term_list = [
            [
                self.mul_terms([str(coeff), "ut_0"]),
                Node("mul", p, [Node(str(k), p), Node("mul", p, [Node("sin", p, [Node("u_0", p)]), Node("ux_0", p)])]),
                self.mul_terms([str(eps / np.pi), "uxx_0"]),
            ]
        ]

        item["tree"] = self.tree_from_list(op_list, term_list)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        CFL = 0.4
        train_num =0
        train_num2 =0
        if p.IC_types.startswith("rarefactiontrain"):
            _, train_num = p.IC_types.split("rarefactiontrain_")
            train_num = int(train_num)
        elif p.IC_types.startswith("rarefaction2train"):
            _, train_num2 = p.IC_types.split("rarefaction2train_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            mode = "periodic"
            numbers = jnp.shape(ICs)[0]
        elif p.IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif p.IC_types == "rarefaction" or (train_num < 3 and train_num > 0) or (
                        train_num2 <= 4 and train_num2 > 0):
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            ends = np.arcsin(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "one_shock" or train_num >= 3 or train_num2 > 4:
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            ends = np.arcsin(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "two_shocks":
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start, p.IC_jump_end, (numbers, 3))
            ends = np.sort(ends, axis=1)
            ends = np.arcsin(ends)
            breakmid1 = rng.uniform(p.x_range * 0.2, p.x_range * 0.5, (numbers,))
            breakmid2 = rng.uniform(p.x_range * 0.5, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 2] * (self.x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        self.x_grid.flatten() < breakmid2[i]) * (
                                 self.x_grid.flatten() >= breakmid1[i]) + ends[i, 0] * (
                                 self.x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        uu = burgers_f(
            p.x_range,
            0.0,
            p.x_num,
            0.0,
            p.t_range / coeff,
            self.dt / coeff,
            p.t_num,
            CFL,
            numbers,
            20,
            rng.randint(100000),
            eps,
            k,
            fluxx="cos",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )
        try:
            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample))
                    res_np.append(sample)

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res

        # for i in range(len(res)):
        #     sns.heatmap(res_np[i][:, :, 0])
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title(
        #         'cos FLUX Equation with coefficient = '+ str(eps)+ "and" + str(k))
        #     plt.show()
        #
        #     this_sample = res_np[i]
        #     print(np.linalg.norm(res_np[i][32:, :, :]) / (
        #            np.linalg.norm(res_np[i][32:, :, :] - np.mean(res_np[i][32:, :, :]))))
        #     # for j in range(p.t_num):
        #     plt.plot(this_sample[0, :])
        #     plt.show()
        #     plt.close()

        return item


    def inviscid_conservation_cosflux_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["sub"]]
        term_list = [
            [
                self.mul_terms([ph,"ut_0"]),
                Node("mul", p, [Node(ph, p), Node("mul", p, [Node("sin", p, [Node("u_0", p)]), Node("ux_0", p)])]),
            ]
        ]
        return op_list, term_list

    def generate_inviscid_conservation_cosflux(self, rng, train, ICs=None,coeff=None):

        p = self.params
        item = {"type": "inviscid_conservation_cosflux"}
        if coeff is not None:
            k = coeff[1]
        else:
            if p.IC_types.startswith("rarefaction") or p.IC_types == "one_shock" or p.IC_types == "two_shocks":
                k = -1
            else:
                k = 1

            k_range = self.get_sample_range(k)
            k = self.refine_floats(rng.uniform(*k_range, (1,)))[0]



        tf = self.tfinals["inviscid_conservation_cosflux"]
        coeff = p.t_range/tf

        op_list = [["sub"]]
        term_list = [
            [
                self.mul_terms([str(coeff), "ut_0"]),
                Node("mul", p, [Node(str(k), p), Node("mul", p, [Node("sin", p, [Node("u_0", p)]), Node("ux_0", p)])]),
            ]
        ]

        item["tree"] = self.tree_from_list(op_list, term_list)

        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation

        CFL = 0.4
        train_num=0
        train_num2 =0
        if p.IC_types.startswith("rarefactiontrain"):
            _, train_num = p.IC_types.split("rarefactiontrain_")
            train_num = int(train_num)
        elif p.IC_types.startswith("rarefaction2train"):
            _, train_num2 = p.IC_types.split("rarefaction2train_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            numbers = jnp.shape(ICs)[0]
            mode = "periodic"
        elif p.IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif p.IC_types == "rarefaction" or (train_num < 4 and train_num > 0) or (
                    train_num2 <= 5 and train_num2 > 0):
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            ends = np.arcsin(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "one_shock" or train_num >= 4 or train_num2 > 5:
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end,(numbers, 2))
            ends = np.sort(ends, axis=1)
            ends = np.arcsin(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = rng.uniform(p.x_range * 0.2, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (self.x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        self.x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif p.IC_types == "two_shocks":
            numbers = num_initial_points * 10
            ends = rng.uniform(p.IC_jump_start,p.IC_jump_end,(numbers, 3))
            ends = np.sort(ends, axis=1)
            ends = np.arcsin(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid1 = rng.uniform(p.x_range * 0.2, p.x_range * 0.5, (numbers,))
            breakmid2 = rng.uniform(p.x_range * 0.5, p.x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = self.x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *p.x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 2] * (self.x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        self.x_grid.flatten() < breakmid2[i]) * (
                                 self.x_grid.flatten() >= breakmid1[i]) + ends[i, 0] * (
                                 self.x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= p.x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        uu = burgers_f(
            p.x_range,
            0.0,
            p.x_num,
            0.0,
            p.t_range / coeff,
            self.dt / coeff,
            p.t_num,
            CFL,
            numbers,
            20,
            rng.randint(100000),
            0,
            k,
            viscous=False,
            fluxx="cos",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )
        try:

            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample))
                    res_np.append(sample)

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res

        # for i in range(len(res)):
        #     sns.heatmap(res_np[i][:, :, 0])
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title(
        #         'cos FLUX Equation with coefficient = ' + "and" + str(k))
        #     plt.show()
        #
        #     this_sample = res_np[i]
        #     print(np.linalg.norm(res_np[i][32:, :, :]) / (
        #         np.linalg.norm(res_np[i][32:, :, :] - np.mean(res_np[i][32:, :, :]))))
        #
        #     # for j in range(p.t_num):
        #     plt.plot(this_sample[0, :])
        #     plt.show()
        #     plt.close()

        return item


    def fplanck_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add", "sub", "sub"]]
        term_list = [
            [
                self.mul_terms([ph, "ut_0"]),
                Node(
                    "mul",
                    p,
                    [
                        Node(ph, p),
                        Node(
                            "mul", p, [Node("cos", p, [Node("mul", p, [Node("u_0", p), Node(ph, p)])]), Node("u_0", p)]
                        ),
                    ],
                ),
                Node(
                    "mul",
                    p,
                    [
                        Node(ph, p),
                        Node(
                            "mul", p, [Node("sin", p, [Node("mul", p, [Node("u_0", p), Node(ph, p)])]), Node("ux_0", p)]
                        ),
                    ],
                ),
                self.mul_terms([ph, "uxx_0"]),
            ]
        ]
        return op_list, term_list

    def generate_fplanck(self, rng, train, ICs=None,coeff = None):
        #
        p = self.params
        tf = self.tfinals["fplanck"]
        um = 1e-6  # micrometer
        viscosity = 1e-3  # viscosity of the medium (Pa·s)
        radius = 0.1 * um  # radius of the particle converted to micrometers
        coeff_t = p.t_range / tf
        # coeff_x = 1/um

        L = 0.1 * um  # Characteristic length scale of the potential, converted to micrometers
        c = 5e-21
        temperature = 300
        item = {"type": "fplanck"}

        viscosity_range = self.get_sample_range(viscosity)
        viscosity = self.refine_floats(rng.uniform(*viscosity_range, (1,)))[0]
        # c_range = self.get_sample_range(c)
        # c = self.refine_floats(rng.uniform(*c_range, (1,)))[0]
        # c = c* 10 **-21
        drag = 6 * np.pi * viscosity * radius  # drag coefficient

        op_list = [["add", "sub", "sub"]]
        term_list = [
            [
                self.mul_terms([str(coeff_t), "ut_0"]),
                Node(
                    "mul",
                    p,
                    [
                        Node(str(c / (drag * L**2)), p),
                        Node(
                            "mul",
                            p,
                            [Node("cos", p, [Node("mul", p, [Node("x", p), Node(str(um / L), p)])]), Node("u_0", p)],
                        ),
                    ],
                ),
                Node(
                    "mul",
                    p,
                    [
                        Node(str(c / (um * drag * L)), p),
                        Node(
                            "mul",
                            p,
                            [Node("sin", p, [Node("mul", p, [Node("x", p), Node(str(um / L), p)])]), Node("ux_0", p)],
                        ),
                    ],
                ),
                self.mul_terms([str(scipy.constants.k * temperature / (drag * um**2)), "uxx_0"]),
            ]
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        # Define the potential function U(x) using micrometers
        U = lambda x: c * np.cos(x / L)

        # Setup the fokker_planck simulation with parameters converted to micrometers
        sim = fokker_planck(
            temperature=temperature,
            drag=drag,
            extent=2 * um,
            # extent converted to micrometers
            resolution=self.dx * um,  # resolution converted to micrometers
            boundary=boundary.periodic,
            potential=U,
        )

        # # Steady-state solution
        # steady = sim.steady_state()
        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        res = []

        for i in range(num_initial_points * 10):
            if p.IC_types == "train":
                mean = rng.uniform(0.5, 1.5)
                std = rng.uniform(0.1, 0.5)
                # pdf = gaussian_pdf(mean * um, std * um)
                p0 = np.exp(-np.square((sim.grid[0] - mean * um) / (std * um)))
                slope = (p0[-1] - p0[0]) / sim.grid[0][-1]
                p0 -= slope * sim.grid[0]
                # p0 = pdf(sim.grid[0])
            else:
                begin = rng.uniform(0.1, 1.0)
                end = rng.uniform(1.1, 1.9)

                def region_func(x):
                    return (x > begin * um) & (x < end * um)

                pdf = uniform_pdf(func=region_func)
                p0 = pdf(sim.grid[0])
            try:
                time, Pt = sim.propagate_interval(p0, tf, Nsteps=p.t_num)
                res.append(torch.from_numpy(Pt).unsqueeze(-1))
                if len(res) >= num_initial_points:
                    break
            except Exception as e:
                pass
        item["data"] = res

        # for i in range(len(res)):
        #     sns.heatmap(np.asarray(res[i])[:, :, 0])
        #     plt.xlabel('x')
        #     plt.ylabel('t')
        #     plt.title('FP')
        #     plt.show()
        #     this_sample = np.asarray(res[i])
        #
        #     for j in range(p.t_num):
        #         plt.plot(this_sample[j,:])
        #     plt.show()
        #     plt.close()

        return item

