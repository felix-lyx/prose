from abc import ABC, abstractmethod

import torch
import numpy as np
from numpy.polynomial import polynomial as P
import copy
from logging import getLogger
from symbolicregression.envs import encoders
from scipy.integrate import solve_ivp

from symbolicregression.envs.node_utils import Node, NodeList, math_constants

logger = getLogger()

operators_real = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "neg": 1,
    # "abs": 1,
    "inv": 1,
    # "sqrt": 1,
    # "log": 1,
    # "exp": 1,
    "sin": 1,
    # "arcsin": 1,
    "cos": 1,
    # "arccos": 1,
    # "tan": 1,
    # "arctan": 1,
    # "pow": 2,
    "pow2": 1,
    # "pow3": 1,
}

operators_extra = dict()


all_operators = {**operators_real, **operators_extra}


# class Node:
#     def __init__(self, value, params, children=None):
#         self.value = value
#         self.children = children if children else []
#         self.params = params

#     def push_child(self, child):
#         self.children.append(child)

#     def prefix(self):
#         s = str(self.value)
#         for c in self.children:
#             s += "," + c.prefix()
#         return s

#     # export to latex qtree format: prefix with \Tree, use package qtree
#     def qtree_prefix(self):
#         s = "[.$" + str(self.value) + "$ "
#         for c in self.children:
#             s += c.qtree_prefix()
#         s += "]"
#         return s

#     def infix(self):
#         nb_children = len(self.children)
#         if nb_children == 0:
#             if self.value.lstrip("-").isdigit():
#                 return str(self.value)
#             else:
#                 s = str(self.value)
#                 return s
#         if nb_children == 1:
#             s = str(self.value)
#             if s == "pow2":
#                 s = "(" + self.children[0].infix() + ")**2"
#             elif s == "pow3":
#                 s = "(" + self.children[0].infix() + ")**3"
#             else:
#                 s = s + "(" + self.children[0].infix() + ")"
#             return s
#         s = "(" + self.children[0].infix()
#         for c in self.children[1:]:
#             s = s + " " + str(self.value) + " " + c.infix()
#         return s + ")"

#     def __len__(self):
#         lenc = 1
#         for c in self.children:
#             lenc += len(c)
#         return lenc

#     def __str__(self):
#         return self.infix()

#     def __repr__(self):
#         return str(self)

#     def val(self, u, deterministic=True):
#         if len(self.children) == 0:
#             if str(self.value).startswith("u_"):
#                 _, dim = self.value.split("_")
#                 dim = int(dim)
#                 return u[:, dim]
#             # elif str(self.value).startswith("x"):
#             #     return x[:, 0]
#             # elif str(self.value) == "rand":
#             #     if deterministic:
#             #         return np.zeros((x.shape[0],))
#             #     return np.random.randn(x.shape[0])
#             elif str(self.value) in math_constants:
#                 return getattr(np, str(self.value)) * np.ones((u.shape[0],))
#             else:
#                 return float(self.value) * np.ones((u.shape[0],))

#         elif self.value == "add":
#             return self.children[0].val(u) + self.children[1].val(u)
#         elif self.value == "sub":
#             return self.children[0].val(u) - self.children[1].val(u)
#         elif self.value == "neg":
#             return -self.children[0].val(u)
#         elif self.value == "mul":
#             m1, m2 = self.children[0].val(u), self.children[1].val(u)
#             try:
#                 return m1 * m2
#             except Exception as e:
#                 # print(e)
#                 nans = np.empty((m1.shape[0],))
#                 nans[:] = np.nan
#                 return nans
#         # if self.value == "pow":
#         #     m1, m2 = self.children[0].val(u), self.children[1].val(u)
#         #     try:
#         #         return np.power(m1, m2)
#         #     except Exception as e:
#         #         # print(e)
#         #         nans = np.empty((m1.shape[0],))
#         #         nans[:] = np.nan
#         #         return nans
#         # if self.value == "max":
#         #     return np.maximum(self.children[0].val(u), self.children[1].val(u))
#         # if self.value == "min":
#         #     return np.minimum(self.children[0].val(u), self.children[1].val(u))
#         elif self.value == "div":
#             denominator = self.children[1].val(u)
#             denominator[denominator == 0.0] = np.nan
#             try:
#                 return self.children[0].val(u) / denominator
#             except Exception as e:
#                 # print(e)
#                 nans = np.empty((denominator.shape[0],))
#                 nans[:] = np.nan
#                 return nans
#         elif self.value == "inv":
#             denominator = self.children[0].val(u)
#             denominator[denominator == 0.0] = np.nan
#             try:
#                 return 1 / denominator
#             except Exception as e:
#                 # print(e)
#                 nans = np.empty((denominator.shape[0],))
#                 nans[:] = np.nan
#                 return nans
#         # if self.value == "log":
#         #     numerator = self.children[0].val(u)
#         #     if self.params.use_abs:
#         #         numerator[numerator <= 0.0] *= -1
#         #     else:
#         #         numerator[numerator <= 0.0] = np.nan
#         #     try:
#         #         return np.log(numerator)
#         #     except Exception as e:
#         #         # print(e)
#         #         nans = np.empty((numerator.shape[0],))
#         #         nans[:] = np.nan
#         #         return nans

#         # if self.value == "sqrt":
#         #     numerator = self.children[0].val(u)
#         #     if self.params.use_abs:
#         #         numerator[numerator <= 0.0] *= -1
#         #     else:
#         #         numerator[numerator < 0.0] = np.nan
#         #     try:
#         #         return np.sqrt(numerator)
#         #     except Exception as e:
#         #         # print(e)
#         #         nans = np.empty((numerator.shape[0],))
#         #         nans[:] = np.nan
#         #         return nans
#         elif self.value == "pow2":
#             numerator = self.children[0].val(u)
#             try:
#                 return numerator**2
#             except Exception as e:
#                 nans = np.empty((numerator.shape[0],))
#                 nans[:] = np.nan
#                 return nans
#         # if self.value == "pow3":
#         #     numerator = self.children[0].val(u)
#         #     try:
#         #         return numerator ** 3
#         #     except Exception as e:
#         #         # print(e)
#         #         nans = np.empty((numerator.shape[0],))
#         #         nans[:] = np.nan
#         #         return nans
#         # if self.value == "abs":
#         #     return np.abs(self.children[0].val(u))
#         # if self.value == "sign":
#         #     return (self.children[0].val(u) >= 0) * 2.0 - 1.0
#         # if self.value == "step":
#         #     u = self.children[0].val(u)
#         #     return u if u > 0 else 0
#         # if self.value == "id":
#         #     return self.children[0].val(u)
#         # if self.value == "fresnel":
#         #     return scipy.special.fresnel(self.children[0].val(u))[0]
#         elif self.value.startswith("eval"):
#             n = self.value[-1]
#             return getattr(scipy.special, self.value[:-1])(n, self.children[0].val(u))[0]
#         else:
#             fn = getattr(np, self.value, None)
#             if fn is not None:
#                 try:
#                     return fn(self.children[0].val(u))
#                 except Exception as e:
#                     nans = np.empty((u.shape[0],))
#                     nans[:] = np.nan
#                     return nans
#             fn = getattr(scipy.special, self.value, None)
#             if fn is not None:
#                 return fn(self.children[0].val(u))
#             assert False, "Could not find function: {}".format(self.value)

#     def replace_node_value(self, old_value, new_value):
#         if self.value == old_value:
#             self.value = new_value
#         for child in self.children:
#             child.replace_node_value(old_value, new_value)


# class NodeList:
#     def __init__(self, nodes):
#         self.nodes = []
#         for node in nodes:
#             self.nodes.append(node)
#         self.params = nodes[0].params

#     def infix(self):
#         return " | ".join([node.infix() for node in self.nodes])

#     def __len__(self):
#         return sum([len(node) for node in self.nodes])

#     def prefix(self):
#         return ",|,".join([node.prefix() for node in self.nodes])

#     def __str__(self):
#         return self.infix()

#     def __repr__(self):
#         return str(self)

#     def val(self, u, deterministic=True):
#         batch_vals = [np.expand_dims(node.val(np.copy(u), deterministic=deterministic), -1) for node in self.nodes]
#         return np.concatenate(batch_vals, -1)

#     def replace_node_value(self, old_value, new_value):
#         for node in self.nodes:
#             node.replace_node_value(old_value, new_value)


class Generator(ABC):
    def __init__(self, params):
        pass

    @abstractmethod
    def generate_datapoints(self, rng):
        pass


class RandomFunctions(Generator):
    def __init__(self, params, special_words):
        super().__init__(params)
        self.params = params
        self.ICs_per_equation = self.params.ICs_per_equation
        self.t_span = [0.0, params.t_range]
        self.t_eval = np.linspace(0.0, params.t_range, params.t_num)
        self.max_int = params.max_int

        self.min_output_dimension = params.min_output_dimension
        self.min_input_dimension = params.min_input_dimension
        self.max_input_dimension = params.max_input_dimension
        self.max_output_dimension = params.max_output_dimension
        self.operators = copy.deepcopy(operators_real)

        self.unaries = [o for o in self.operators.keys() if np.abs(self.operators[o]) == 1]

        self.binaries = [o for o in self.operators.keys() if np.abs(self.operators[o]) == 2]

        self.constants = [str(i) for i in range(-self.max_int, self.max_int + 1) if i != 0]
        self.constants += math_constants
        self.variables = (
            ["rand"]
            + [f"u_{i}" for i in range(self.max_output_dimension)]
            + [f"x_{i}" for i in range(self.max_input_dimension)]
        )
        self.symbols = (
            list(self.operators)
            + self.constants
            + self.variables
            + ["|", "INT+", "INT-", "FLOAT+", "FLOAT-", "pow", "0"]
        )
        self.constants.remove("CONSTANT")

        self.general_encoder = encoders.GeneralEncoder(params, self.symbols, all_operators)
        self.float_encoder = self.general_encoder.float_encoder
        self.float_words = special_words + sorted(list(set(self.float_encoder.symbols)))
        self.equation_encoder = self.general_encoder.equation_encoder
        self.equation_words = sorted(list(set(self.symbols)))
        self.equation_words = special_words + self.equation_words
        self.ode_generator = ODEGenerator(
            self.params,
            self.float_encoder,
            self.equation_encoder,
            self.t_span,
            self.t_eval,
        )

    def generate_float(self, rng, exponent=None):
        sign = rng.choice([-1, 1])
        mantissa = float(rng.choice(range(1, 10**self.params.float_precision)))
        min_power = -self.params.max_exponent_prefactor - (self.params.float_precision + 1) // 2
        max_power = self.params.max_exponent_prefactor - (self.params.float_precision + 1) // 2
        if not exponent:
            exponent = rng.randint(min_power, max_power + 1)
        constant = sign * (mantissa * 10**exponent)
        return str(constant)

    def generate_int(self, rng):
        return str(rng.choice(self.constants + self.extra_constants))

    def generate_one_sample(self, rng, train=True, type=None):
        return self.ode_generator.generate_sample(rng, train=train, type=type)

    def generate_datapoints(self, rng):
        raise NotImplementedError


class ODEGenerator:
    def __init__(self, params, float_encoder, equation_encoder, t_span, t_eval):
        self.params = params
        self.float_encoder = float_encoder
        self.equation_encoder = equation_encoder
        self.t_span = t_span
        self.t_eval = t_eval

        self.ICs_per_equation = params.ICs_per_equation
        self.eval_ICs_per_equation = self.ICs_per_equation // 5
        self.rtol = 1e-5
        self.atol = 1e-6

        self.ph = "<PLACEHOLDER>"
        self.tree_skeletons = dict()

        self.type_to_dim = {
            "thomas": 3,
            "lorenz_3d": 3,
            "aizawa": 3,
            "chen_lee": 3,
            "dadras": 3,
            "rossler": 3,
            "halvorsen": 3,
            "fabrikant": 3,
            "sprott_B": 3,
            "sprott_linz_F": 3,
            "four_wing": 3,
            "lorenz_96_4d": 4,
            "lorenz_96_5d": 5,
            "duffing": 3,
            "double_pendulum": 4,
        }

        if self.params.types == "chaotic_ode_3d":
            self.types = [
                "thomas",
                "lorenz_3d",
                "aizawa",
                "chen_lee",
                "dadras",
                "rossler",
                "halvorsen",
                "fabrikant",
                "sprott_B",
                "sprott_linz_F",
                "four_wing",
                "duffing",
            ]
        elif self.params.types == "chaotic_ode_all":
            self.types = [
                "thomas",
                "lorenz_3d",
                "aizawa",
                "chen_lee",
                "dadras",
                "rossler",
                "halvorsen",
                "fabrikant",
                "sprott_B",
                "sprott_linz_F",
                "four_wing",
                "lorenz_96_4d",
                "lorenz_96_5d",
                "duffing",
                "double_pendulum",
            ]
        elif self.params.types == "scalar_ode":
            self.types = ["trig", "poly"]
        else:
            try:
                self.types = self.params.types.split(",")
                assert len(self.types) > 0
            except:
                assert False, "invalid type: {}".format(self.params.types)

        self.cur_idx = 0
        self.total_types = len(self.types)

        if self.params.noisy_text_input:
            p = self.params
            self.missing_locations = dict()
            self.addition_locations = dict()

            # generate terms to be added (polynomials of degree at most 2)
            self.addition_terms = dict()
            for dim in range(self.params.min_output_dimension, self.params.max_output_dimension + 1):
                cur_addition_terms = [Node(self.ph, p)]

                for i in range(dim):
                    cur_addition_terms.append(Node("mul", p, [Node(self.ph, p), Node(f"u_{i}", p)]))

                    for j in range(i, dim):
                        cur_addition_terms.append(
                            Node(
                                "mul",
                                p,
                                [
                                    Node(self.ph, p),
                                    Node("mul", p, [Node(f"u_{i}", p), Node(f"u_{j}", p)]),
                                ],
                            )
                        )
                self.addition_terms[dim] = cur_addition_terms

    def get_sample_range(self, mean):
        """
        Generate interval for sample parameters
        """
        gamma = self.params.ode_param_range_gamma
        half_range = np.abs(mean) * gamma
        return [mean - half_range, mean + half_range]

    def get_skeleton_tree(self, type, mode=0, rng=None):
        """
        Generate skeleton tree for text input, with possibly added/deleted terms
        """
        if mode == 0:
            # no text noise
            if type not in self.tree_skeletons:
                op_list, term_list = getattr(self, type + "_tree_list")()
                tree = self.tree_from_list(op_list, term_list)
                tree_skeleton = self.equation_encoder.encode_with_placeholder(tree)
                self.tree_skeletons[type] = tree_skeleton
            return self.tree_skeletons[type]
        elif mode == -1:
            # term deletion
            assert rng is not None
            if type == "double_pendulum":
                return self.double_pendulum_missing_term(type, rng)
            else:
                return self.tree_with_missing_term(type, rng)
        elif mode == 1:
            # term addition
            assert rng is not None
            if type == "double_pendulum":
                return self.double_pendulum_additional_term(type, rng)
            else:
                return self.tree_with_additional_term(type, rng)
        else:
            assert False, "Unknown mode {}".format(mode)

    def tree_with_additional_term(self, type, rng):
        op_list, term_list = getattr(self, type + "_tree_list")()
        term_to_add = rng.choice(self.addition_terms[self.type_to_dim[type]])

        if type not in self.addition_locations:
            coords = []
            for i in range(len(term_list)):
                lst = term_list[i]
                for j in range(len(lst) + 1):
                    coords.append((i, j))
            self.addition_locations[type] = coords

        coords = self.addition_locations[type]
        coord = coords[rng.choice(len(coords))]
        i, j = coord[0], coord[1]
        if j == 0:
            if term_list[i][0].value == "neg":
                term_list[i][0] = term_list[i][0].children[0]
                op_list[i].insert(0, "sub")
                term_list[i].insert(0, term_to_add)
            else:
                op_list[i].insert(0, "add")
                term_list[i].insert(0, term_to_add)
        else:
            op_list[i].insert(j - 1, "add")
            term_list[i].insert(j, term_to_add)
        tree = self.tree_from_list(op_list, term_list)
        return self.equation_encoder.encode_with_placeholder(tree)

    def tree_with_missing_term(self, type, rng):
        op_list, term_list = getattr(self, type + "_tree_list")()

        if type not in self.missing_locations:
            coords = []
            for i in range(len(term_list)):
                lst = term_list[i]
                if len(lst) <= 1:
                    continue
                for j in range(len(lst)):
                    coords.append((i, j))
            self.missing_locations[type] = coords

        coords = self.missing_locations[type]
        coord = coords[rng.choice(len(coords))]
        i, j = coord[0], coord[1]

        if j == 0:
            op = op_list[i].pop(j)
            term_list[i].pop(j)
            if op == "sub":
                term = term_list[i][0]
                term_list[i][0] = Node("neg", self.params, [term])
        else:
            op = op_list[i].pop(j - 1)
            term_list[i].pop(j)

        tree = self.tree_from_list(op_list, term_list)
        return self.equation_encoder.encode_with_placeholder(tree)

    def refine_floats(self, lst):
        """
        Refine floats to specified precision
        """
        return np.array(self.float_encoder.decode(self.float_encoder.encode(lst)))

    def poly_tree(self, degree, var, params=None):
        """
        Generate a tree containing a polynomial with given degree and variable
        """
        assert degree >= 1
        tree = Node(var, params)
        for _ in range(degree - 1):
            tree = Node("mul", params, [Node(var, params), tree])
        return tree

    def mul_terms(self, lst):
        """
        Generate a tree containing multiplication of terms in lst
        """
        p = self.params
        tree = None
        for i in reversed(range(len(lst))):
            if tree is None:
                tree = Node(lst[i], p)
            else:
                tree = Node("mul", p, [Node(lst[i], p), tree])
        return tree

    def add_terms(self, lst):
        """
        Generate a tree containing addition of terms in lst
        """
        p = self.params
        tree = None
        for i in reversed(range(len(lst))):
            if tree is None:
                tree = lst[i]
            else:
                tree = Node("add", p, [lst[i], tree])
        return tree

    def tree_from_list(self, op_list, term_list):
        """
        Generate a tree from the operator list and term list
        """
        p = self.params
        res = []
        dim = len(op_list)
        for i in range(dim):
            ops = op_list[i]
            terms = term_list[i]
            assert len(ops) + 1 == len(terms)
            tree = None
            for j in range(len(terms)):
                if tree is None:
                    tree = terms[j]
                else:
                    tree = Node(ops[j - 1], p, [tree, terms[j]])
            res.append(tree)

        return NodeList(res)

    def generate_sample(self, rng, train=True, type=None):
        """
        Generate a tree sample
        """
        if type is None:
            type = self.types[self.cur_idx]
            self.cur_idx = (self.cur_idx + 1) % self.total_types

        item = getattr(self, "generate_" + type)(rng, train)

        return item

    def generate_trig(self, rng, train):
        # parameters
        max_terms = 4
        max_trig_degree = 4
        weight_range = [-2.0, 2.0]

        # generate tree
        item = {"type": "trig"}
        terms = rng.randint(1, max_terms + 1)
        degrees = rng.uniform(1, max_trig_degree, terms)  # rng.randint(1, max_trig_degree, terms)
        weights = rng.uniform(*weight_range, terms)
        degrees = self.refine_floats(degrees)
        weights = self.refine_floats(weights)
        sin_or_cos = rng.choice(["sin", "cos"], terms)

        tree = None
        for j in range(terms):
            cur_tree = Node(
                sin_or_cos[j],
                self.params,
                [
                    Node(
                        "mul",
                        self.params,
                        [Node(str(degrees[j]), self.params), Node("u_0", self.params)],
                    )
                ],
            )
            cur_tree = Node("mul", self.params, [Node(str(weights[j]), self.params), cur_tree])
            tree = cur_tree if tree is None else Node("add", self.params, [tree, cur_tree])
        item["tree"] = NodeList([tree])

        def f(weights, sin_or_cos, degrees, terms):
            trigs = {"sin": np.sin, "cos": np.cos}
            return lambda u: np.stack([weights[k] * trigs[sin_or_cos[k]](degrees[k] * u) for k in range(terms)]).sum(
                axis=0
            )

        item["func"] = f(weights, sin_or_cos, degrees, terms)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-10, 10, num_initial_points * 10)

        res = []
        fun = lambda _, y: item["func"](y)
        for y_0 in y_0s:
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    [y_0],
                    method="BDF",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.flatten().astype(np.single)))
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def generate_poly(self, rng, train):
        # parameters
        max_poly_degree = 2
        weight_range = [-2.0, 2.0]
        root_range = [-3.0, 2.5]

        # generate tree
        item = {"type": "poly"}
        degree = rng.randint(1, max_poly_degree + 1)
        weight = rng.uniform(*weight_range)
        roots = rng.uniform(*root_range, degree)
        coeffs = P.polyfromroots(roots) * weight
        coeffs = self.refine_floats(coeffs)
        tree = None
        for j in range(len(coeffs)):
            if j == 0:
                cur_tree = Node(str(coeffs[j]), self.params)
            else:
                cur_tree = Node(
                    "mul",
                    self.params,
                    [
                        Node(str(coeffs[j]), self.params),
                        self.poly_tree(j, "u_0", self.params),
                    ],
                )
            tree = cur_tree if tree is None else Node("add", self.params, [tree, cur_tree])
        item["tree"] = NodeList([tree])
        item["func"] = P.Polynomial(coeffs)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-10, 10, num_initial_points * 10)

        res = []
        fun = lambda _, y: item["func"](y)
        for y_0 in y_0s:
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    [y_0],
                    method="BDF",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.flatten().astype(np.single)))
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def thomas_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add"], ["add"], ["add"]]
        term_list = [
            [
                Node("sin", p, [Node("u_1", p)]),
                Node("mul", p, [Node(ph, p), Node("u_0", p)]),
            ],
            [
                Node("sin", p, [Node("u_2", p)]),
                Node("mul", p, [Node(ph, p), Node("u_1", p)]),
            ],
            [
                Node("sin", p, [Node("u_0", p)]),
                Node("mul", p, [Node(ph, p), Node("u_2", p)]),
            ],
        ]
        return op_list, term_list

    def generate_thomas(self, rng, train):
        # parameters
        b_range = self.get_sample_range(0.17)  # [0.15, 0.195]

        # generate tree
        item = {"type": "thomas"}
        p = self.params
        b = self.refine_floats(rng.uniform(*b_range, (1,)))[0]
        b_str = str(-b)

        op_list = [["add"], ["add"], ["add"]]
        term_list = [
            [
                Node("sin", p, [Node("u_1", p)]),
                Node("mul", p, [Node(b_str, p), Node("u_0", p)]),
            ],
            [
                Node("sin", p, [Node("u_2", p)]),
                Node("mul", p, [Node(b_str, p), Node("u_1", p)]),
            ],
            [
                Node("sin", p, [Node("u_0", p)]),
                Node("mul", p, [Node(b_str, p), Node("u_2", p)]),
            ],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(b):
            def f(u):
                u0, u1, u2 = u[0], u[1], u[2]
                s = np.sin(u)
                return np.array([s[1] - b * u0, s[2] - b * u1, s[0] - b * u2]).reshape(u.shape)

            return f

        item["func"] = f_closure(b)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 3))

        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="BDF",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def lorenz_3d_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add"], ["sub", "sub"], ["add"]]
        term_list = [
            [self.mul_terms([ph, "u_1"]), self.mul_terms([ph, "u_0"])],
            [
                self.mul_terms([ph, "u_0"]),
                self.mul_terms(["u_0", "u_2"]),
                self.mul_terms(["u_1"]),
            ],
            [self.mul_terms(["u_0", "u_1"]), self.mul_terms([ph, "u_2"])],
        ]
        return op_list, term_list

    def generate_lorenz_3d(self, rng, train):
        # parameters
        sigma_range = self.get_sample_range(10)  # [9.5, 10.5]
        beta_range = self.get_sample_range(8 / 3)  # [2.4, 2.8]
        rho_range = self.get_sample_range(28)  # [27.0, 29.0]

        # generate tree
        item = {"type": "lorenz_3d"}
        sigma = self.refine_floats(rng.uniform(*sigma_range, (1,)))[0]
        rho = self.refine_floats(rng.uniform(*rho_range, (1,)))[0]
        beta = self.refine_floats(rng.uniform(*beta_range, (1,)))[0]
        p = self.params

        op_list = [["add"], ["sub", "sub"], ["add"]]
        term_list = [
            [self.mul_terms([str(sigma), "u_1"]), self.mul_terms([str(-sigma), "u_0"])],
            [
                self.mul_terms([str(rho), "u_0"]),
                self.mul_terms(["u_0", "u_2"]),
                self.mul_terms(["u_1"]),
            ],
            [self.mul_terms(["u_0", "u_1"]), self.mul_terms([str(-beta), "u_2"])],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(sigma, rho, beta):
            def f(u):
                u0, u1, u2 = u[0], u[1], u[2]
                return np.array([sigma * (u1 - u0), u0 * (rho - u2) - u1, u0 * u1 - beta * u2]).reshape(u.shape)

            return f

        item["func"] = f_closure(sigma, rho, beta)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 3))

        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="BDF",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def aizawa_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add", "add"], ["add", "add"], ["add", "sub", "sub", "add"]]
        term_list = [
            [
                self.mul_terms(["u_0", "u_2"]),
                self.mul_terms([ph, "u_0"]),
                self.mul_terms([ph, "u_1"]),
            ],
            [
                self.mul_terms([ph, "u_0"]),
                self.mul_terms(["u_1", "u_2"]),
                self.mul_terms([ph, "u_1"]),
            ],
            [
                self.mul_terms([ph]),
                self.mul_terms([ph, "u_2"]),
                Node(
                    "div",
                    p,
                    [self.poly_tree(3, "u_2", p), Node(ph, p)],
                ),
                self.mul_terms(["u_0", "u_0"]),
                self.mul_terms([ph, "u_0", "u_0", "u_0", "u_2"]),
            ],
        ]
        return op_list, term_list

    def generate_aizawa(self, rng, train):
        # parameters
        a_range = self.get_sample_range(0.95)  # [0.93, 0.96]
        b_range = self.get_sample_range(0.7)  # [0.65, 0.75]
        c_range = self.get_sample_range(0.6)  # [0.55, 0.65]
        d_range = self.get_sample_range(3.5)  # [3.3, 3.7]
        e_range = self.get_sample_range(0.25)  # [0.23, 0.27]
        f_range = self.get_sample_range(0.1)  # [0.08, 0.12]

        # generate tree
        item = {"type": "aizawa"}
        a = self.refine_floats(rng.uniform(*a_range, (1,)))[0]
        b = self.refine_floats(rng.uniform(*b_range, (1,)))[0]
        c = self.refine_floats(rng.uniform(*c_range, (1,)))[0]
        d = self.refine_floats(rng.uniform(*d_range, (1,)))[0]
        e = self.refine_floats(rng.uniform(*e_range, (1,)))[0]
        f = self.refine_floats(rng.uniform(*f_range, (1,)))[0]

        p = self.params

        op_list = [["add", "add"], ["add", "add"], ["add", "sub", "sub", "add"]]
        term_list = [
            [
                self.mul_terms(["u_0", "u_2"]),
                self.mul_terms([str(-b), "u_0"]),
                self.mul_terms([str(-d), "u_1"]),
            ],
            [
                self.mul_terms([str(d), "u_0"]),
                self.mul_terms(["u_1", "u_2"]),
                self.mul_terms([str(-b), "u_1"]),
            ],
            [
                self.mul_terms([str(c)]),
                self.mul_terms([str(a), "u_2"]),
                Node(
                    "div",
                    p,
                    [self.poly_tree(3, "u_2", p), Node("3.0", p)],
                ),
                self.mul_terms(["u_0", "u_0"]),
                self.mul_terms([str(f), "u_0", "u_0", "u_0", "u_2"]),
            ],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(a, b, c, d, e, f):
            def g(u):
                u0, u1, u2 = u[0], u[1], u[2]
                return np.array(
                    [
                        (u2 - b) * u0 - d * u1,
                        d * u0 + (u2 - b) * u1,
                        c + a * u2 - u2 * u2 * u2 / 3 - u0 * u0 + f * u2 * u0 * u0 * u0,
                    ]
                ).reshape(u.shape)

            return g

        item["func"] = f_closure(a, b, c, d, e, f)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 3))

        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="BDF",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def chen_lee_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["sub"], ["add"], ["add"]]
        term_list = [
            [self.mul_terms([ph, "u_0"]), self.mul_terms(["u_1", "u_2"])],
            [self.mul_terms([ph, "u_1"]), self.mul_terms(["u_0", "u_2"])],
            [
                self.mul_terms([ph, "u_2"]),
                Node(
                    "div",
                    p,
                    [
                        Node("mul", p, [Node("u_0", p), Node("u_1", p)]),
                        Node(ph, p),
                    ],
                ),
            ],
        ]
        return op_list, term_list

    def generate_chen_lee(self, rng, train):
        # parameters
        a_range = self.get_sample_range(5)  # [4.9, 5.1]
        d_range = self.get_sample_range(-0.38)  # [-0.36, -0.4]

        # generate tree
        item = {"type": "chen_lee"}
        a = self.refine_floats(rng.uniform(*a_range, (1,)))[0]
        d = self.refine_floats(rng.uniform(*d_range, (1,)))[0]
        b = -10.0

        p = self.params

        op_list = [["sub"], ["add"], ["add"]]
        term_list = [
            [self.mul_terms([str(a), "u_0"]), self.mul_terms(["u_1", "u_2"])],
            [self.mul_terms([str(b), "u_1"]), self.mul_terms(["u_0", "u_2"])],
            [
                self.mul_terms([str(d), "u_2"]),
                Node(
                    "div",
                    p,
                    [
                        Node("mul", p, [Node("u_0", p), Node("u_1", p)]),
                        Node("3.0", p),
                    ],
                ),
            ],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(a, b, d):
            def f(u):
                u0, u1, u2 = u[0], u[1], u[2]
                return np.array([a * u0 - u1 * u2, b * u1 + u0 * u2, d * u2 + u0 * u1 / 3]).reshape(u.shape)

            return f

        item["func"] = f_closure(a, b, d)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 3))

        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="LSODA",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def dadras_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add", "add"], ["add", "add"], ["add"]]
        term_list = [
            [
                self.mul_terms([ph, "u_1"]),
                self.mul_terms([ph, "u_0"]),
                self.mul_terms([ph, "u_1", "u_2"]),
            ],
            [
                self.mul_terms([ph, "u_1"]),
                self.mul_terms([ph, "u_0", "u_2"]),
                self.mul_terms([ph, "u_2"]),
            ],
            [self.mul_terms([ph, "u_0", "u_1"]), self.mul_terms([ph, "u_2"])],
        ]
        return op_list, term_list

    def generate_dadras(self, rng, train):
        # parameters
        a_range = self.get_sample_range(1.25)  # [2.0, 3.0]
        b_range = self.get_sample_range(1.15)  # [1.9, 2.7]
        c_range = self.get_sample_range(0.75)  # [1.3, 1.7]
        d_range = self.get_sample_range(0.8)  # [1.2, 2.0]
        e_range = self.get_sample_range(4.0)  # [7.0, 9.0]

        # generate tree
        item = {"type": "dadras"}
        a = self.refine_floats(rng.uniform(*a_range, (1,)))[0]
        b = self.refine_floats(rng.uniform(*b_range, (1,)))[0]
        c = self.refine_floats(rng.uniform(*c_range, (1,)))[0]
        d = self.refine_floats(rng.uniform(*d_range, (1,)))[0]
        e = self.refine_floats(rng.uniform(*e_range, (1,)))[0]

        p = self.params

        op_list = [["add", "add"], ["add", "add"], ["add"]]
        term_list = [
            [
                self.mul_terms([str(0.5), "u_1"]),
                self.mul_terms([str(-a), "u_0"]),
                self.mul_terms([str(b), "u_1", "u_2"]),
            ],
            [
                self.mul_terms([str(c), "u_1"]),
                self.mul_terms([str(-0.5), "u_0", "u_2"]),
                self.mul_terms([str(0.5), "u_2"]),
            ],
            [self.mul_terms([str(d), "u_0", "u_1"]), self.mul_terms([str(-e), "u_2"])],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(a, b, c, d, e):
            def f(u):
                u0, u1, u2 = u[0], u[1], u[2]
                return np.array(
                    [
                        0.5 * u1 - a * u0 + b * u1 * u2,
                        c * u1 - 0.5 * u0 * u2 + 0.5 * u2,
                        d * u0 * u1 - e * u2,
                    ]
                ).reshape(u.shape)

            return f

        item["func"] = f_closure(a, b, c, d, e)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 3))

        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="LSODA",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def rossler_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["sub"], ["add"], ["add", "add"]]
        term_list = [
            [Node("neg", p, [Node("u_1", p)]), self.mul_terms(["u_2"])],
            [self.mul_terms(["u_0"]), self.mul_terms([ph, "u_1"])],
            [
                self.mul_terms([ph]),
                self.mul_terms(["u_0", "u_2"]),
                self.mul_terms([ph, "u_2"]),
            ],
        ]
        return op_list, term_list

    def generate_rossler(self, rng, train):
        # parameters
        a_range = self.get_sample_range(0.1)  # [0.09, 0.11]
        b_range = self.get_sample_range(0.1)  # [0.09, 0.11]
        c_range = self.get_sample_range(14)  # [13.8, 14.2]

        # generate tree
        item = {"type": "rossler"}
        a = self.refine_floats(rng.uniform(*a_range, (1,)))[0]
        b = self.refine_floats(rng.uniform(*b_range, (1,)))[0]
        c = self.refine_floats(rng.uniform(*c_range, (1,)))[0]

        p = self.params

        op_list = [["sub"], ["add"], ["add", "add"]]
        term_list = [
            [Node("neg", p, [Node("u_1", p)]), self.mul_terms(["u_2"])],
            [self.mul_terms(["u_0"]), self.mul_terms([str(a), "u_1"])],
            [
                self.mul_terms([str(b)]),
                self.mul_terms(["u_0", "u_2"]),
                self.mul_terms([str(-c), "u_2"]),
            ],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(a, b, c):
            def f(u):
                u0, u1, u2 = u[0], u[1], u[2]
                return np.array([-u1 - u2, u0 + a * u1, b + u2 * (u0 - c)]).reshape(u.shape)

            return f

        item["func"] = f_closure(a, b, c)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 3))

        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="LSODA",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def halvorsen_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["sub", "sub", "sub"], ["sub", "sub", "sub"], ["sub", "sub", "sub"]]
        term_list = [
            [
                self.mul_terms([ph, "u_0"]),
                self.mul_terms(["u_1"]),
                self.mul_terms(["u_2"]),
                self.mul_terms([ph, "u_1", "u_1"]),
            ],
            [
                self.mul_terms([ph, "u_1"]),
                self.mul_terms(["u_2"]),
                self.mul_terms(["u_0"]),
                self.mul_terms([ph, "u_2", "u_2"]),
            ],
            [
                self.mul_terms([ph, "u_2"]),
                self.mul_terms(["u_0"]),
                self.mul_terms(["u_1"]),
                self.mul_terms([ph, "u_0", "u_0"]),
            ],
        ]
        return op_list, term_list

    def generate_halvorsen(self, rng, train):
        # parameters
        a_range = self.get_sample_range(-0.35)

        # generate tree
        item = {"type": "halvorsen"}
        a = self.refine_floats(rng.uniform(*a_range, (1,)))[0]

        p = self.params

        op_list = [["sub", "sub", "sub"], ["sub", "sub", "sub"], ["sub", "sub", "sub"]]
        term_list = [
            [
                self.mul_terms([str(a), "u_0"]),
                self.mul_terms(["u_1"]),
                self.mul_terms(["u_2"]),
                self.mul_terms([str(0.25), "u_1", "u_1"]),
            ],
            [
                self.mul_terms([str(a), "u_1"]),
                self.mul_terms(["u_2"]),
                self.mul_terms(["u_0"]),
                self.mul_terms([str(0.25), "u_2", "u_2"]),
            ],
            [
                self.mul_terms([str(a), "u_2"]),
                self.mul_terms(["u_0"]),
                self.mul_terms(["u_1"]),
                self.mul_terms([str(0.25), "u_0", "u_0"]),
            ],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(a):
            def f(u):
                u0, u1, u2 = u[0], u[1], u[2]
                return np.array(
                    [
                        a * u0 - u1 - u2 - 0.25 * u1 * u1,
                        a * u1 - u2 - u0 - 0.25 * u2 * u2,
                        a * u2 - u0 - u1 - 0.25 * u0 * u0,
                    ]
                ).reshape(u.shape)

            return f

        item["func"] = f_closure(a)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 3))

        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="RK45",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def fabrikant_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["sub", "add", "add"], ["add", "sub", "add"], ["add"]]
        term_list = [
            [
                self.mul_terms(["u_1", "u_2"]),
                self.mul_terms(["u_1"]),
                self.mul_terms(["u_0", "u_0", "u_1"]),
                self.mul_terms([ph, "u_0"]),
            ],
            [
                self.mul_terms([ph, "u_0", "u_2"]),
                self.mul_terms(["u_0"]),
                self.mul_terms(["u_0", "u_0", "u_0"]),
                self.mul_terms([ph, "u_1"]),
            ],
            [self.mul_terms([ph, "u_2"]), self.mul_terms([ph, "u_0", "u_1", "u_2"])],
        ]
        return op_list, term_list

    def generate_fabrikant(self, rng, train):
        # parameters
        alpha_range = self.get_sample_range(0.98)  # [0.9, 1.0]
        gamma_range = self.get_sample_range(0.1)  # [0.05, 0.15]

        # generate tree
        item = {"type": "fabrikant"}
        alpha = self.refine_floats(-2.0 * rng.uniform(*alpha_range, (1,)))[0]
        gamma = self.refine_floats(rng.uniform(*gamma_range, (1,)))[0]

        p = self.params

        op_list = [["sub", "add", "add"], ["add", "sub", "add"], ["add"]]
        term_list = [
            [
                self.mul_terms(["u_1", "u_2"]),
                self.mul_terms(["u_1"]),
                self.mul_terms(["u_0", "u_0", "u_1"]),
                self.mul_terms([str(gamma), "u_0"]),
            ],
            [
                self.mul_terms(["3.0", "u_0", "u_2"]),
                self.mul_terms(["u_0"]),
                self.mul_terms(["u_0", "u_0", "u_0"]),
                self.mul_terms([str(gamma), "u_1"]),
            ],
            [
                self.mul_terms([str(alpha), "u_2"]),
                self.mul_terms(["-2.0", "u_0", "u_1", "u_2"]),
            ],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(alpha, gamma):
            def f(u):
                u0, u1, u2 = u[0], u[1], u[2]
                return np.array(
                    [
                        u1 * (u2 - 1 + u0 * u0) + gamma * u0,
                        u0 * (3 * u2 + 1 - u0 * u0) + gamma * u1,
                        alpha * u2 - 2 * u0 * u1 * u2,
                    ]
                ).reshape(u.shape)

            return f

        item["func"] = f_closure(alpha, gamma)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 3))

        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="BDF",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e5):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def sprott_B_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [[], ["add"], ["sub"]]
        term_list = [
            [self.mul_terms([ph, "u_1", "u_2"])],
            [self.mul_terms(["u_0"]), self.mul_terms([ph, "u_1"])],
            [self.mul_terms([ph]), self.mul_terms(["u_0", "u_1"])],
        ]
        return op_list, term_list

    def generate_sprott_B(self, rng, train):
        # parameters
        a_range = self.get_sample_range(0.4)  # [0.35, 0.45]
        b_range = self.get_sample_range(1.2)  # [1.15, 1.25]
        c_range = self.get_sample_range(1)  # [0.95, 1.05]

        # generate tree
        item = {"type": "sprott_B"}
        a = self.refine_floats(rng.uniform(*a_range, (1,)))[0]
        b = self.refine_floats(rng.uniform(*b_range, (1,)))[0]
        c = self.refine_floats(rng.uniform(*c_range, (1,)))[0]

        p = self.params

        op_list = [[], ["add"], ["sub"]]
        term_list = [
            [self.mul_terms([str(a), "u_1", "u_2"])],
            [self.mul_terms(["u_0"]), self.mul_terms([str(-b), "u_1"])],
            [self.mul_terms([str(c)]), self.mul_terms(["u_0", "u_1"])],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(a, b, c):
            def f(u):
                u0, u1, u2 = u[0], u[1], u[2]
                return np.array([a * u1 * u2, u0 - b * u1, c - u0 * u1]).reshape(u.shape)

            return f

        item["func"] = f_closure(a, b, c)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 3))

        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="BDF",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def sprott_linz_F_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add"], ["sub"], ["sub"]]
        term_list = [
            [self.mul_terms(["u_1"]), self.mul_terms(["u_2"])],
            [self.mul_terms([ph, "u_1"]), self.mul_terms(["u_0"])],
            [self.mul_terms(["u_0", "u_0"]), self.mul_terms(["u_2"])],
        ]
        return op_list, term_list

    def generate_sprott_linz_F(self, rng, train):
        # parameters
        a_range = self.get_sample_range(0.5)  # [0.4, 0.6]

        # generate tree
        item = {"type": "sprott_linz_F"}
        a = self.refine_floats(rng.uniform(*a_range, (1,)))[0]

        p = self.params

        op_list = [["add"], ["sub"], ["sub"]]
        term_list = [
            [self.mul_terms(["u_1"]), self.mul_terms(["u_2"])],
            [self.mul_terms([str(a), "u_1"]), self.mul_terms(["u_0"])],
            [self.mul_terms(["u_0", "u_0"]), self.mul_terms(["u_2"])],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(a):
            def f(u):
                u0, u1, u2 = u[0], u[1], u[2]
                return np.array([u1 + u2, -u0 + a * u1, u0 * u0 - u2]).reshape(u.shape)

            return f

        item["func"] = f_closure(a)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 3))

        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="BDF",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def generate_three_scroll_unified(self, rng, train):
        # parameters
        a_range = [32.2, 32.6]
        b_range = [45.6, 46.0]
        c_range = [1.15, 1.21]
        d_range = [0.1, 0.16]
        e_range = [0.5, 0.65]
        f_range = [14.5, 14.9]

        # generate tree
        item = {"type": "three_scroll_unified"}
        a = self.refine_floats(rng.uniform(*a_range, (1,)))[0]
        b = self.refine_floats(rng.uniform(*b_range, (1,)))[0]
        c = self.refine_floats(rng.uniform(*c_range, (1,)))[0]
        d = self.refine_floats(rng.uniform(*d_range, (1,)))[0]
        e = self.refine_floats(rng.uniform(*e_range, (1,)))[0]
        f = self.refine_floats(rng.uniform(*f_range, (1,)))[0]

        p = self.params
        tree1 = Node(
            "add",
            p,
            [
                Node(
                    "mul",
                    p,
                    [Node(str(a), p), Node("sub", p, [Node("u_1", p), Node("u_0", p)])],
                ),
                Node(
                    "mul",
                    p,
                    [Node("mul", p, [Node(str(d), p), Node("u_0", p)]), Node("u_2", p)],
                ),
            ],
        )
        tree2 = Node(
            "add",
            p,
            [
                Node("mul", p, [Node(str(f), p), Node("u_1", p)]),
                Node(
                    "sub",
                    p,
                    [
                        Node("mul", p, [Node(str(b), p), Node("u_0", p)]),
                        Node("mul", p, [Node("u_0", p), Node("u_2", p)]),
                    ],
                ),
            ],
        )
        tree3 = Node(
            "sub",
            p,
            [
                Node(
                    "add",
                    p,
                    [
                        Node("mul", p, [Node(str(c), p), Node("u_2", p)]),
                        Node("mul", p, [Node("u_0", p), Node("u_1", p)]),
                    ],
                ),
                Node(
                    "mul",
                    p,
                    [Node("mul", p, [Node(str(e), p), Node("u_0", p)]), Node("u_0", p)],
                ),
            ],
        )

        item["tree"] = NodeList([tree1, tree2, tree3])

        def f_closure(a, b, c, d, e, f):
            def g(u):
                u0, u1, u2 = u[0], u[1], u[2]
                return np.array(
                    [
                        a * (u1 - u0) + d * u0 * u2,
                        b * u0 - u0 * u2 + f * u1,
                        c * u2 + u0 * u1 - e * u0 * u0,
                    ]
                ).reshape(u.shape)

            return g

        item["func"] = f_closure(a, b, c, d, e, f)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 3))

        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="BDF",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def four_wing_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [["add"], ["add", "sub"], ["sub"]]
        term_list = [
            [self.mul_terms([ph, "u_0"]), self.mul_terms(["u_1", "u_2"])],
            [
                self.mul_terms([ph, "u_0"]),
                self.mul_terms([ph, "u_1"]),
                self.mul_terms(["u_0", "u_2"]),
            ],
            [Node("neg", p, [Node("u_2", p)]), self.mul_terms(["u_0", "u_1"])],
        ]
        return op_list, term_list

    def generate_four_wing(self, rng, train):
        # parameters
        a_range = self.get_sample_range(0.2)  # [0.18, 0.22]
        b_range = self.get_sample_range(0.01)  # [0.0, 0.02]
        c_range = self.get_sample_range(-0.4)  # [-0.45, -0.35]

        # generate tree
        item = {"type": "four_wing"}
        a = self.refine_floats(rng.uniform(*a_range, (1,)))[0]
        b = self.refine_floats(rng.uniform(*b_range, (1,)))[0]
        c = self.refine_floats(rng.uniform(*c_range, (1,)))[0]

        p = self.params

        op_list = [["add"], ["add", "sub"], ["sub"]]
        term_list = [
            [self.mul_terms([str(a), "u_0"]), self.mul_terms(["u_1", "u_2"])],
            [
                self.mul_terms([str(b), "u_0"]),
                self.mul_terms([str(c), "u_1"]),
                self.mul_terms(["u_0", "u_2"]),
            ],
            [Node("neg", p, [Node("u_2", p)]), self.mul_terms(["u_0", "u_1"])],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(a, b, c):
            def f(u):
                u0, u1, u2 = u[0], u[1], u[2]
                return np.array([a * u0 + u1 * u2, b * u0 + c * u1 - u0 * u2, -u2 - u0 * u1]).reshape(u.shape)

            return f

        item["func"] = f_closure(a, b, c)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 3))

        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="BDF",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def lorenz_96_4d_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [
            ["sub", "sub", "add"],
            ["sub", "sub", "add"],
            ["sub", "sub", "add"],
            ["sub", "sub", "add"],
        ]
        term_list = [
            [
                self.mul_terms(["u_1", "u_3"]),
                self.mul_terms(["u_2", "u_3"]),
                Node("u_0", p),
                Node(ph, p),
            ],
            [
                self.mul_terms(["u_0", "u_2"]),
                self.mul_terms(["u_0", "u_3"]),
                Node("u_1", p),
                Node(ph, p),
            ],
            [
                self.mul_terms(["u_1", "u_3"]),
                self.mul_terms(["u_0", "u_1"]),
                Node("u_2", p),
                Node(ph, p),
            ],
            [
                self.mul_terms(["u_0", "u_2"]),
                self.mul_terms(["u_1", "u_2"]),
                Node("u_3", p),
                Node(ph, p),
            ],
        ]
        return op_list, term_list

    def generate_lorenz_96_4d(self, rng, train):
        # parameters
        F_range = self.get_sample_range(8)  # [7.0, 9.0]

        # generate tree
        item = {"type": "lorenz_96_4d"}
        F = self.refine_floats(rng.uniform(*F_range, (1,)))[0]

        p = self.params

        op_list = [
            ["sub", "sub", "add"],
            ["sub", "sub", "add"],
            ["sub", "sub", "add"],
            ["sub", "sub", "add"],
        ]
        term_list = [
            [
                self.mul_terms(["u_1", "u_3"]),
                self.mul_terms(["u_2", "u_3"]),
                Node("u_0", p),
                Node(str(F), p),
            ],
            [
                self.mul_terms(["u_0", "u_2"]),
                self.mul_terms(["u_0", "u_3"]),
                Node("u_1", p),
                Node(str(F), p),
            ],
            [
                self.mul_terms(["u_1", "u_3"]),
                self.mul_terms(["u_0", "u_1"]),
                Node("u_2", p),
                Node(str(F), p),
            ],
            [
                self.mul_terms(["u_0", "u_2"]),
                self.mul_terms(["u_1", "u_2"]),
                Node("u_3", p),
                Node(str(F), p),
            ],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(F):
            def f(u):
                u0, u1, u2, u3 = u[0], u[1], u[2], u[3]
                return np.array(
                    [
                        (u1 - u2) * u3 - u0 + F,
                        (u2 - u3) * u0 - u1 + F,
                        (u3 - u0) * u1 - u2 + F,
                        (u0 - u1) * u2 - u3 + F,
                    ]
                ).reshape(u.shape)

            return f

        item["func"] = f_closure(F)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 4))

        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="BDF",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def lorenz_96_5d_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [
            ["sub", "sub", "add"],
            ["sub", "sub", "add"],
            ["sub", "sub", "add"],
            ["sub", "sub", "add"],
            ["sub", "sub", "add"],
        ]
        term_list = [
            [
                self.mul_terms(["u_1", "u_4"]),
                self.mul_terms(["u_3", "u_4"]),
                Node("u_0", p),
                Node(ph, p),
            ],
            [
                self.mul_terms(["u_0", "u_2"]),
                self.mul_terms(["u_0", "u_4"]),
                Node("u_1", p),
                Node(ph, p),
            ],
            [
                self.mul_terms(["u_1", "u_3"]),
                self.mul_terms(["u_0", "u_1"]),
                Node("u_2", p),
                Node(ph, p),
            ],
            [
                self.mul_terms(["u_2", "u_4"]),
                self.mul_terms(["u_1", "u_2"]),
                Node("u_3", p),
                Node(ph, p),
            ],
            [
                self.mul_terms(["u_0", "u_3"]),
                self.mul_terms(["u_2", "u_3"]),
                Node("u_4", p),
                Node(ph, p),
            ],
        ]
        return op_list, term_list

    def generate_lorenz_96_5d(self, rng, train):
        # parameters
        F_range = self.get_sample_range(8)  # [7.0, 9.0]

        # generate tree
        item = {"type": "lorenz_96_5d"}
        F = self.refine_floats(rng.uniform(*F_range, (1,)))[0]

        p = self.params

        op_list = [
            ["sub", "sub", "add"],
            ["sub", "sub", "add"],
            ["sub", "sub", "add"],
            ["sub", "sub", "add"],
            ["sub", "sub", "add"],
        ]
        term_list = [
            [
                self.mul_terms(["u_1", "u_4"]),
                self.mul_terms(["u_3", "u_4"]),
                Node("u_0", p),
                Node(str(F), p),
            ],
            [
                self.mul_terms(["u_0", "u_2"]),
                self.mul_terms(["u_0", "u_4"]),
                Node("u_1", p),
                Node(str(F), p),
            ],
            [
                self.mul_terms(["u_1", "u_3"]),
                self.mul_terms(["u_0", "u_1"]),
                Node("u_2", p),
                Node(str(F), p),
            ],
            [
                self.mul_terms(["u_2", "u_4"]),
                self.mul_terms(["u_1", "u_2"]),
                Node("u_3", p),
                Node(str(F), p),
            ],
            [
                self.mul_terms(["u_0", "u_3"]),
                self.mul_terms(["u_2", "u_3"]),
                Node("u_4", p),
                Node(str(F), p),
            ],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(F):
            def f(u):
                u0, u1, u2, u3, u4 = u[0], u[1], u[2], u[3], u[4]
                return np.array(
                    [
                        (u1 - u3) * u4 - u0 + F,
                        (u2 - u4) * u0 - u1 + F,
                        (u3 - u0) * u1 - u2 + F,
                        (u4 - u1) * u2 - u3 + F,
                        (u0 - u2) * u3 - u4 + F,
                    ]
                ).reshape(u.shape)

            return f

        item["func"] = f_closure(F)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 5))

        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="BDF",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )
                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def duffing_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [[], [], ["add", "add", "add"]]
        term_list = [
            [
                Node(ph, p),
            ],
            [
                Node("u_2", p),
            ],
            [
                self.mul_terms([ph, "u_2"]),
                self.mul_terms([ph, "u_1"]),
                self.mul_terms([ph, "u_1", "u_1", "u_1"]),
                Node("mul", p, [Node(ph, p), Node("cos", p, [self.mul_terms([ph, "u_0"])])]),
            ],
        ]
        return op_list, term_list

    def generate_duffing(self, rng, train):
        # parameters
        alpha_range = self.get_sample_range(-1)  # [-1.2, -0.8]
        beta_range = self.get_sample_range(-5)  # [-5.5, -4.5]
        gamma_range = self.get_sample_range(8)  # [7.0, 9.0]
        delta_range = self.get_sample_range(-0.02)  # [-0.025, -0.015]
        omega_range = self.get_sample_range(0.5)  # [0.45, 0.55]

        # generate tree
        item = {"type": "duffing"}
        alpha = self.refine_floats(rng.uniform(*alpha_range, (1,)))[0]
        beta = self.refine_floats(rng.uniform(*beta_range, (1,)))[0]
        gamma = self.refine_floats(rng.uniform(*gamma_range, (1,)))[0]
        delta = self.refine_floats(rng.uniform(*delta_range, (1,)))[0]
        omega = self.refine_floats(rng.uniform(*omega_range, (1,)))[0]

        p = self.params

        op_list = [[], [], ["add", "add", "add"]]
        term_list = [
            [
                Node("1.0", p),
            ],
            [
                Node("u_2", p),
            ],
            [
                self.mul_terms([str(delta), "u_2"]),
                self.mul_terms([str(alpha), "u_1"]),
                self.mul_terms([str(beta), "u_1", "u_1", "u_1"]),
                Node("mul", p, [Node(str(gamma), p), Node("cos", p, [self.mul_terms([str(omega), "u_0"])])]),
            ],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(alpha, beta, gamma, delta, omega):
            def f(u):
                u0, u1, u2 = u[0], u[1], u[2]
                return np.array(
                    [
                        1.0,
                        u2,
                        delta * u2 + alpha * u1 + beta * u1 * u1 * u1 + gamma * np.cos(omega * u0),
                    ]
                ).reshape(u.shape)

            return f

        item["func"] = f_closure(alpha, beta, gamma, delta, omega)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 3))
        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="BDF",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )
                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def double_pendulum_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [[], [], ["div"], ["div"]]
        term_list = [
            [
                Node("u_2", p),
            ],
            [
                Node("u_3", p),
            ],
            [
                self.add_terms(
                    [
                        Node("mul", p, [Node(ph, p), Node("sin", p, [Node("u_0", p)])]),
                        Node(
                            "mul",
                            p,
                            [
                                Node(ph, p),
                                Node(
                                    "sin",
                                    p,
                                    [Node("add", p, [Node("u_0", p), Node("mul", p, [Node(ph, p), Node("u_1", p)])])],
                                ),
                            ],
                        ),
                        Node(
                            "mul",
                            p,
                            [
                                Node(ph, p),
                                Node(
                                    "mul",
                                    p,
                                    [
                                        Node("sin", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])]),
                                        Node(
                                            "add",
                                            p,
                                            [
                                                self.mul_terms(["u_3", "u_3"]),
                                                Node(
                                                    "mul",
                                                    p,
                                                    [
                                                        self.mul_terms(["u_2", "u_2"]),
                                                        Node(
                                                            "cos", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])]
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ]
                ),
                Node(
                    "sub",
                    p,
                    [
                        Node(ph, p),
                        Node(
                            "cos", p, [Node("mul", p, [Node(ph, p), Node("sub", p, [Node("u_0", p), Node("u_1", p)])])]
                        ),
                    ],
                ),
            ],
            [
                Node(
                    "mul",
                    p,
                    [
                        Node("sin", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])]),
                        self.add_terms(
                            [
                                self.mul_terms([ph, "u_2", "u_2"]),
                                Node("mul", p, [Node(ph, p), Node("cos", p, [Node("u_0", p)])]),
                                Node(
                                    "mul",
                                    p,
                                    [
                                        self.mul_terms(["u_3", "u_3"]),
                                        Node("cos", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])]),
                                    ],
                                ),
                            ]
                        ),
                    ],
                ),
                Node(
                    "sub",
                    p,
                    [
                        Node(ph, p),
                        Node(
                            "cos", p, [Node("mul", p, [Node(ph, p), Node("sub", p, [Node("u_0", p), Node("u_1", p)])])]
                        ),
                    ],
                ),
            ],
        ]
        return op_list, term_list

    def generate_double_pendulum(self, rng, train):
        # parameters
        g_range = self.get_sample_range(9.81)  # [9.75, 9.85]
        l_range = self.get_sample_range(1)  # [0.9, 1.1]

        # generate tree
        item = {"type": "double_pendulum"}
        g = rng.uniform(*g_range)
        l = rng.uniform(*l_range)

        a = self.refine_floats(np.array([-3 * g / l]))[0]
        b = self.refine_floats(np.array([-g / l]))[0]
        c = self.refine_floats(np.array([4 * g / l]))[0]

        p = self.params

        op_list = [[], [], ["div"], ["div"]]
        term_list = [
            [
                Node("u_2", p),
            ],
            [
                Node("u_3", p),
            ],
            [
                self.add_terms(
                    [
                        Node("mul", p, [Node(str(a), p), Node("sin", p, [Node("u_0", p)])]),
                        Node(
                            "mul",
                            p,
                            [
                                Node(str(b), p),
                                Node(
                                    "sin",
                                    p,
                                    [
                                        Node(
                                            "add",
                                            p,
                                            [Node("u_0", p), Node("mul", p, [Node("-2.0", p), Node("u_1", p)])],
                                        )
                                    ],
                                ),
                            ],
                        ),
                        Node(
                            "mul",
                            p,
                            [
                                Node("-2.0", p),
                                Node(
                                    "mul",
                                    p,
                                    [
                                        Node("sin", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])]),
                                        Node(
                                            "add",
                                            p,
                                            [
                                                self.mul_terms(["u_3", "u_3"]),
                                                Node(
                                                    "mul",
                                                    p,
                                                    [
                                                        self.mul_terms(["u_2", "u_2"]),
                                                        Node(
                                                            "cos", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])]
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ]
                ),
                Node(
                    "sub",
                    p,
                    [
                        Node("3.0", p),
                        Node(
                            "cos",
                            p,
                            [Node("mul", p, [Node("2.0", p), Node("sub", p, [Node("u_0", p), Node("u_1", p)])])],
                        ),
                    ],
                ),
            ],
            [
                Node(
                    "mul",
                    p,
                    [
                        Node("sin", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])]),
                        self.add_terms(
                            [
                                self.mul_terms(["4.0", "u_2", "u_2"]),
                                Node("mul", p, [Node(str(c), p), Node("cos", p, [Node("u_0", p)])]),
                                Node(
                                    "mul",
                                    p,
                                    [
                                        self.mul_terms(["u_3", "u_3"]),
                                        Node("cos", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])]),
                                    ],
                                ),
                            ]
                        ),
                    ],
                ),
                Node(
                    "sub",
                    p,
                    [
                        Node("3.0", p),
                        Node(
                            "cos",
                            p,
                            [Node("mul", p, [Node("2.0", p), Node("sub", p, [Node("u_0", p), Node("u_1", p)])])],
                        ),
                    ],
                ),
            ],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(a, b, c):
            def f(u):
                u0, u1, u2, u3 = u[0], u[1], u[2], u[3]
                return np.array(
                    [
                        u2,
                        u3,
                        (
                            a * np.sin(u0)
                            + b * np.sin(u0 - 2 * u1)
                            - 2 * np.sin(u0 - u1) * (u3 * u3 + u2 * u2 * np.cos(u0 - u1))
                        )
                        / (3 - np.cos(2 * (u0 - u1))),
                        np.sin(u0 - u1)
                        * (4 * u2 * u2 + c * np.cos(u0) + u3 * u3 * np.cos(u0 - u1))
                        / (3 - np.cos(2 * (u0 - u1))),
                    ]
                ).reshape(u.shape)

            return f

        item["func"] = f_closure(a, b, c)

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 4))
        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="BDF",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )
                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item

    def double_pendulum_additional_term(self, type, rng):
        p = self.params
        ph = self.ph
        add_list0 = [
            Node("mul", p, [Node(ph, p), Node("sin", p, [Node("u_0", p)])]),
            Node(
                "mul",
                p,
                [
                    Node(ph, p),
                    Node("sin", p, [Node("add", p, [Node("u_0", p), Node("mul", p, [Node(ph, p), Node("u_1", p)])])]),
                ],
            ),
            Node(
                "mul",
                p,
                [
                    Node(ph, p),
                    Node(
                        "mul",
                        p,
                        [
                            Node("sin", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])]),
                            Node(
                                "add",
                                p,
                                [
                                    self.mul_terms(["u_3", "u_3"]),
                                    Node(
                                        "mul",
                                        p,
                                        [
                                            self.mul_terms(["u_2", "u_2"]),
                                            Node("cos", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])]),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ]
        add_list1 = [
            self.mul_terms([ph, "u_2", "u_2"]),
            Node("mul", p, [Node(ph, p), Node("cos", p, [Node("u_0", p)])]),
            Node(
                "mul",
                p,
                [self.mul_terms(["u_3", "u_3"]), Node("cos", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])])],
            ),
        ]

        term_to_add = rng.choice(self.addition_terms[self.type_to_dim[type]])

        if type not in self.addition_locations:
            coords = []
            for j in range(len(add_list0) + 1):
                coords.append((0, j))
            for j in range(len(add_list1) + 1):
                coords.append((1, j))
            self.addition_locations[type] = coords

        coords = self.addition_locations[type]
        coord = coords[rng.choice(len(coords))]
        i, j = coord[0], coord[1]
        if i == 0:
            add_list0.insert(j, term_to_add)
        else:
            add_list1.insert(j, term_to_add)

        op_list = [[], [], ["div"], ["div"]]
        term_list = [
            [
                Node("u_2", p),
            ],
            [
                Node("u_3", p),
            ],
            [
                self.add_terms(add_list0),
                Node(
                    "sub",
                    p,
                    [
                        Node(ph, p),
                        Node(
                            "cos", p, [Node("mul", p, [Node(ph, p), Node("sub", p, [Node("u_0", p), Node("u_1", p)])])]
                        ),
                    ],
                ),
            ],
            [
                Node(
                    "mul",
                    p,
                    [Node("sin", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])]), self.add_terms(add_list1)],
                ),
                Node(
                    "sub",
                    p,
                    [
                        Node(ph, p),
                        Node(
                            "cos", p, [Node("mul", p, [Node(ph, p), Node("sub", p, [Node("u_0", p), Node("u_1", p)])])]
                        ),
                    ],
                ),
            ],
        ]
        tree = self.tree_from_list(op_list, term_list)
        return self.equation_encoder.encode_with_placeholder(tree)

    def double_pendulum_missing_term(self, type, rng):
        p = self.params
        ph = self.ph
        add_list0 = [
            Node("mul", p, [Node(ph, p), Node("sin", p, [Node("u_0", p)])]),
            Node(
                "mul",
                p,
                [
                    Node(ph, p),
                    Node("sin", p, [Node("add", p, [Node("u_0", p), Node("mul", p, [Node(ph, p), Node("u_1", p)])])]),
                ],
            ),
            Node(
                "mul",
                p,
                [
                    Node(ph, p),
                    Node(
                        "mul",
                        p,
                        [
                            Node("sin", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])]),
                            Node(
                                "add",
                                p,
                                [
                                    self.mul_terms(["u_3", "u_3"]),
                                    Node(
                                        "mul",
                                        p,
                                        [
                                            self.mul_terms(["u_2", "u_2"]),
                                            Node("cos", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])]),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ]
        add_list1 = [
            self.mul_terms([ph, "u_2", "u_2"]),
            Node("mul", p, [Node(ph, p), Node("cos", p, [Node("u_0", p)])]),
            Node(
                "mul",
                p,
                [self.mul_terms(["u_3", "u_3"]), Node("cos", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])])],
            ),
        ]

        if type not in self.missing_locations:
            coords = []
            for j in range(len(add_list0)):
                coords.append((0, j))
            for j in range(len(add_list1)):
                coords.append((1, j))
            self.missing_locations[type] = coords

        coords = self.missing_locations[type]
        coord = coords[rng.choice(len(coords))]
        i, j = coord[0], coord[1]

        if i == 0:
            add_list0.pop(j)
        else:
            add_list1.pop(j)
        op_list = [[], [], ["div"], ["div"]]
        term_list = [
            [
                Node("u_2", p),
            ],
            [
                Node("u_3", p),
            ],
            [
                self.add_terms(add_list0),
                Node(
                    "sub",
                    p,
                    [
                        Node(ph, p),
                        Node(
                            "cos", p, [Node("mul", p, [Node(ph, p), Node("sub", p, [Node("u_0", p), Node("u_1", p)])])]
                        ),
                    ],
                ),
            ],
            [
                Node(
                    "mul",
                    p,
                    [Node("sin", p, [Node("sub", p, [Node("u_0", p), Node("u_1", p)])]), self.add_terms(add_list1)],
                ),
                Node(
                    "sub",
                    p,
                    [
                        Node(ph, p),
                        Node(
                            "cos", p, [Node("mul", p, [Node(ph, p), Node("sub", p, [Node("u_0", p), Node("u_1", p)])])]
                        ),
                    ],
                ),
            ],
        ]
        tree = self.tree_from_list(op_list, term_list)
        return self.equation_encoder.encode_with_placeholder(tree)

    def glycolysis_7d_tree_list(self):
        p = self.params
        ph = self.ph
        op_list = [[], [], ["add", "add", "add"]]
        term_list = [
            [
                Node(ph, p),
            ],
            [
                Node("u_2", p),
            ],
            [
                self.mul_terms([ph, "u_2"]),
                self.mul_terms([ph, "u_1"]),
                self.mul_terms([ph, "u_1", "u_1", "u_1"]),
                Node("mul", p, [Node(ph, p), Node("cos", p, [self.mul_terms([ph, "u_0"])])]),
            ],
        ]
        return op_list, term_list

    def generate_glycolysis_7d(self, rng, train):
        # parameters
        c1_range = [2.3, 2.7]
        c3_range = [13.1, 13.9]
        d3_range = [-6.5, -5.5]
        e4_range = [15.0, 17.0]
        f2_range = [-13.5, -12.5]
        g2_range = [-3.4, -2.8]
        h5_range = [-33.0, -31.0]
        j2_range = [-19.0, -17.0]

        # generate tree
        item = {"type": "glycolysis_7d"}
        c1 = self.refine_floats(rng.uniform(*c1_range, (1,)))[0]
        c2 = -100.0
        c3 = self.refine_floats(rng.uniform(*c3_range, (1,)))[0]
        d1 = self.refine_floats(np.array([-2 * c2]))[0]
        d2 = c3
        d3 = self.refine_floats(rng.uniform(*d3_range, (1,)))[0]
        d4 = -d3
        e1 = -d3
        e2 = -64.0
        e3 = e1
        e4 = self.refine_floats(rng.uniform(*e4_range, (1,)))[0]
        f1 = -e2
        f2 = self.refine_floats(rng.uniform(*f2_range, (1,)))[0]
        f3 = -f2
        f4 = e4
        f5 = c2
        g1 = self.refine_floats(np.array([-f2 / 10]))[0]
        g2 = self.refine_floats(rng.uniform(*g2_range, (1,)))[0]
        h1 = -d1
        h2 = c3
        h3 = 128.0
        h4 = self.refine_floats(np.array([-h3 / 100]))[0]
        h5 = self.refine_floats(rng.uniform(*h5_range, (1,)))[0]
        j1 = e1
        j2 = self.refine_floats(rng.uniform(*j2_range, (1,)))[0]
        j3 = c2

        p = self.params

        op_list = [
            ["add"],
            ["add", "add"],
            ["add", "add", "add", "add"],
            ["add", "add", "add", "add"],
            ["add"],
            ["add", "add", "add"],
            ["add", "add"],
        ]
        term_list = [
            [
                Node(str(c1), p),
                Node(
                    "div",
                    p,
                    [
                        self.mul_terms([str(c2), "u_0", "u_5"]),
                        Node("add", p, [Node("1.0", p), self.mul_terms([str(c3), "u_5", "u_5", "u_5", "u_5"])]),
                    ],
                ),
            ],
            [
                Node(
                    "div",
                    p,
                    [
                        self.mul_terms([str(d1), "u_0", "u_5"]),
                        Node("add", p, [Node("1.0", p), self.mul_terms([str(d2), "u_5", "u_5", "u_5", "u_5"])]),
                    ],
                ),
                self.mul_terms([str(d3), "u_1"]),
                self.mul_terms([str(d4), "u_1", "u_6"]),
            ],
            [
                self.mul_terms([str(e1), "u_1"]),
                self.mul_terms([str(e2), "u_2"]),
                self.mul_terms([str(e3), "u_1", "u_6"]),
                self.mul_terms([str(e4), "u_2", "u_5"]),
                self.mul_terms([str(f5), "u_3", "u_6"]),
            ],
            [
                self.mul_terms([str(f1), "u_2"]),
                self.mul_terms([str(e2), "u_3"]),
                self.mul_terms([str(f3), "u_4"]),
                self.mul_terms([str(f4), "u_2", "u_5"]),
                self.mul_terms([str(f5), "u_3", "u_6"]),
            ],
            [
                self.mul_terms([str(g1), "u_0"]),
                self.mul_terms([str(g2), "u_4"]),
            ],
            [
                self.mul_terms([str(h3), "u_2"]),
                self.mul_terms([str(h5), "u_5"]),
                self.mul_terms([str(h4), "u_2", "u_5"]),
                Node(
                    "div",
                    p,
                    [
                        self.mul_terms([str(h1), "u_0", "u_5"]),
                        Node("add", p, [Node("1.0", p), self.mul_terms([str(h2), "u_5", "u_5", "u_5", "u_5"])]),
                    ],
                ),
            ],
            [
                self.mul_terms([str(j1), "u_1"]),
                self.mul_terms([str(j2), "u_1", "u_6"]),
                self.mul_terms([str(j3), "u_3", "u_6"]),
            ],
        ]
        item["tree"] = self.tree_from_list(op_list, term_list)

        def f_closure(
            c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, f1, f2, f3, f4, f5, g1, g2, h1, h2, h3, h4, h5, j1, j2, j3
        ):
            def f(u):
                u0, u1, u2, u3, u4, u5, u6 = u[0], u[1], u[2], u[3], u[4], u[5], u[6]
                return np.array(
                    [
                        c1 + (c2 * u0 * u5) / (1 + c3 * (u5**4)),
                        (d1 * u0 * u5) / (1 + d2 * (u5**4)) + d3 * u1 + d4 * u1 * u6,
                        e1 * u1 + e2 * u2 + e3 * u1 * u6 + e4 * u2 * u5 + f5 * u3 * u6,
                        f1 * u2 + e2 * u3 + f3 * u4 + f4 * u2 * u5 + f5 * u3 * u6,
                        g1 * u0 + g2 * u4,
                        h3 * u2 + h5 * u5 + h4 * u2 * u5 + (h1 * u0 * u5) / (1 + h2 * (u5**4)),
                        j1 * u1 + j2 * u1 * u6 + j3 * u3 * u6,
                    ]
                ).reshape(u.shape)

            return f

        item["func"] = f_closure(
            c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, f1, f2, f3, f4, f5, g1, g2, h1, h2, h3, h4, h5, j1, j2, j3
        )

        # ODE solve
        num_initial_points = self.ICs_per_equation if train else self.eval_ICs_per_equation
        y_0s = rng.uniform(-2, 2, (num_initial_points * 10, 7))
        res = []
        fun = lambda _, y: item["func"](y)
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    self.t_span,
                    y_0,
                    method="BDF",
                    t_eval=self.t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )
                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))  # (t_num, 3)
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        return item
