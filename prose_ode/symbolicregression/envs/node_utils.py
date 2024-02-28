import numpy as np
import scipy.special

# math_constants = ["e", "pi", "euler_gamma", "CONSTANT"]
math_constants = ["CONSTANT"]


class Node:
    def __init__(self, value, params, children=None):
        self.value = value
        self.children = children if children else []
        self.params = params

    def push_child(self, child):
        self.children.append(child)

    def prefix(self):
        s = str(self.value)
        for c in self.children:
            s += "," + c.prefix()
        return s

    # export to latex qtree format: prefix with \Tree, use package qtree
    def qtree_prefix(self):
        s = "[.$" + str(self.value) + "$ "
        for c in self.children:
            s += c.qtree_prefix()
        s += "]"
        return s

    def infix(self):
        nb_children = len(self.children)
        if nb_children == 0:
            if self.value.lstrip("-").isdigit():
                return str(self.value)
            else:
                s = str(self.value)
                return s
        if nb_children == 1:
            s = str(self.value)
            if s == "pow2":
                s = "(" + self.children[0].infix() + ")**2"
            elif s == "pow3":
                s = "(" + self.children[0].infix() + ")**3"
            else:
                s = s + "(" + self.children[0].infix() + ")"
            return s
        s = "(" + self.children[0].infix()
        for c in self.children[1:]:
            s = s + " " + str(self.value) + " " + c.infix()
        return s + ")"

    def __len__(self):
        lenc = 1
        for c in self.children:
            lenc += len(c)
        return lenc

    def __str__(self):
        return self.infix()

    def __repr__(self):
        return str(self)

    def val(self, u, deterministic=True):
        if len(self.children) == 0:
            if str(self.value).startswith("u_"):
                _, dim = self.value.split("_")
                dim = int(dim)
                return u[:, dim]
            # elif str(self.value).startswith("x"):
            #     return x[:, 0]
            # elif str(self.value) == "rand":
            #     if deterministic:
            #         return np.zeros((x.shape[0],))
            #     return np.random.randn(x.shape[0])
            elif str(self.value) in math_constants:
                return getattr(np, str(self.value)) * np.ones((u.shape[0],))
            else:
                return float(self.value) * np.ones((u.shape[0],))

        elif self.value == "add":
            return self.children[0].val(u) + self.children[1].val(u)
        elif self.value == "sub":
            return self.children[0].val(u) - self.children[1].val(u)
        elif self.value == "neg":
            return -self.children[0].val(u)
        elif self.value == "mul":
            m1, m2 = self.children[0].val(u), self.children[1].val(u)
            try:
                return m1 * m2
            except Exception as e:
                # print(e)
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        # if self.value == "pow":
        #     m1, m2 = self.children[0].val(u), self.children[1].val(u)
        #     try:
        #         return np.power(m1, m2)
        #     except Exception as e:
        #         # print(e)
        #         nans = np.empty((m1.shape[0],))
        #         nans[:] = np.nan
        #         return nans
        # if self.value == "max":
        #     return np.maximum(self.children[0].val(u), self.children[1].val(u))
        # if self.value == "min":
        #     return np.minimum(self.children[0].val(u), self.children[1].val(u))
        elif self.value == "div":
            denominator = self.children[1].val(u)
            denominator[denominator == 0.0] = np.nan
            try:
                return self.children[0].val(u) / denominator
            except Exception as e:
                # print(e)
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans
        elif self.value == "inv":
            denominator = self.children[0].val(u)
            denominator[denominator == 0.0] = np.nan
            try:
                return 1 / denominator
            except Exception as e:
                # print(e)
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans
        # if self.value == "log":
        #     numerator = self.children[0].val(u)
        #     if self.params.use_abs:
        #         numerator[numerator <= 0.0] *= -1
        #     else:
        #         numerator[numerator <= 0.0] = np.nan
        #     try:
        #         return np.log(numerator)
        #     except Exception as e:
        #         # print(e)
        #         nans = np.empty((numerator.shape[0],))
        #         nans[:] = np.nan
        #         return nans

        # if self.value == "sqrt":
        #     numerator = self.children[0].val(u)
        #     if self.params.use_abs:
        #         numerator[numerator <= 0.0] *= -1
        #     else:
        #         numerator[numerator < 0.0] = np.nan
        #     try:
        #         return np.sqrt(numerator)
        #     except Exception as e:
        #         # print(e)
        #         nans = np.empty((numerator.shape[0],))
        #         nans[:] = np.nan
        #         return nans
        elif self.value == "pow2":
            numerator = self.children[0].val(u)
            try:
                return numerator**2
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        # if self.value == "pow3":
        #     numerator = self.children[0].val(u)
        #     try:
        #         return numerator ** 3
        #     except Exception as e:
        #         # print(e)
        #         nans = np.empty((numerator.shape[0],))
        #         nans[:] = np.nan
        #         return nans
        # if self.value == "abs":
        #     return np.abs(self.children[0].val(u))
        # if self.value == "sign":
        #     return (self.children[0].val(u) >= 0) * 2.0 - 1.0
        # if self.value == "step":
        #     u = self.children[0].val(u)
        #     return u if u > 0 else 0
        # if self.value == "id":
        #     return self.children[0].val(u)
        # if self.value == "fresnel":
        #     return scipy.special.fresnel(self.children[0].val(u))[0]
        elif self.value.startswith("eval"):
            n = self.value[-1]
            return getattr(scipy.special, self.value[:-1])(n, self.children[0].val(u))[0]
        else:
            fn = getattr(np, self.value, None)
            if fn is not None:
                try:
                    return fn(self.children[0].val(u))
                except Exception as e:
                    nans = np.empty((u.shape[0],))
                    nans[:] = np.nan
                    return nans
            fn = getattr(scipy.special, self.value, None)
            if fn is not None:
                return fn(self.children[0].val(u))
            assert False, "Could not find function: {}".format(self.value)

    def replace_node_value(self, old_value, new_value):
        if self.value == old_value:
            self.value = new_value
        for child in self.children:
            child.replace_node_value(old_value, new_value)


class NodeList:
    def __init__(self, nodes):
        self.nodes = []
        for node in nodes:
            self.nodes.append(node)
        self.params = nodes[0].params

    def infix(self):
        return " | ".join([node.infix() for node in self.nodes])

    def __len__(self):
        return sum([len(node) for node in self.nodes])

    def prefix(self):
        return ",|,".join([node.prefix() for node in self.nodes])

    def __str__(self):
        return self.infix()

    def __repr__(self):
        return str(self)

    def val(self, u, deterministic=True):
        batch_vals = [np.expand_dims(node.val(np.copy(u), deterministic=deterministic), -1) for node in self.nodes]
        return np.concatenate(batch_vals, -1)

    def replace_node_value(self, old_value, new_value):
        for node in self.nodes:
            node.replace_node_value(old_value, new_value)
