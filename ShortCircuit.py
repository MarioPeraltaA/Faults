"""Fault currents.

This program calculates the short circuit current sequences for
the following kind of faults:

    1. Single Line-to-Ground
    2. Line-to-Line
    3. Double Line-to-Ground
    4. Three-Phase fault.

Author: Mario R. Peralta A.
Electric Power and Energy Research Laboratory (EPERLab).

"""
class sysData:
    def __init__(self):
        self.generators = {}
        self.transformers = {}
        self.lines = {}

    def call_data(self, path: str) -> None:
        import pandas as pd
        # Read data
        df = pd.read_excel(path,
                        sheet_name=None)
        # List of all sheets
        sheets = list(df.keys())

        # Set data regarding the sheet
        for sheet in df.keys():
            # Generators
            if sheet == sheets[0]:
                for c in df[sheet].columns:
                    cols = [v for v in df[sheet][c]]
                    # Update attribute
                    self.generators[c] = cols
            # Transformers
            elif sheet == sheets[1]:
                for c in df[sheet].columns:
                    cols = [v for v in df[sheet][c]]
                    # Update attribute
                    self.transformers[c] = cols
            # Lines
            elif sheet == sheets[2]:
                for c in df[sheet].columns:
                    cols = [v for v in df[sheet][c]]
                    # Update attribute
                    self.lines[c] = cols

class Generator:
    pass

class Transformer:
    pass


class Conductor:
    def __init__(self):
        self.R0_pu = None
        self.R1_pu = None
        self.R2_pu = self.R1_pu
        self.X0_pu = None
        self.X1_pu = None
        self.X2_pu = self.X1_pu

import numpy as np

class Bus:

    def __init__(self,
                 V: float,
                 deg: float,
                 PL: float,
                 QL: float,
                 G: float,
                 B: float,
                 Vb: float,
                 bus_type: str):
        self.V = V
        self.theta = deg*np.pi / 180      # To rad
        self.PL = PL
        self.QL = QL
        self.G = G     # Conductance of compensation
        self.B = B     # Susceptance of compensation
        self.Vb = Vb
        self.bus_type = bus_type

    def get_phasor(self) -> complex:
        return self.V * np.exp(1j*self.theta)

class Line:
    """Gegenal type of conductors.

    Represents fundamentally the pi model of a
    conductors or transformer.
    """
    def __init__(self,
                 from_bus: Bus,
                 to_bus: Bus,
                 R_pu: float,
                 X_pu: float,
                 from_Y: complex,
                 to_Y: complex):
        self.from_bus = from_bus
        self.to_bus = to_bus
        self.R_pu = R_pu
        self.X_pu = X_pu
        self.from_Y = from_Y   # Half shunt
        self.to_Y = to_Y       # Other half shunt

class System:
    r"""Network.

    ``System`` class represents the electric network.

    It makes up an auxiliary list in order to organize the buses.
    """
    def __init__(self):
        self.slack = None
        self.buses = []
        self.non_slack_buses = []
        self.PQ_buses = []
        self.PV_buses = []
        self.lines = []

    def add_generators(self, data: dict) -> None:
        list_keys = list(data.keys())
        # Number of generators
        nG = len(data[list_keys[0]])
        generators = [Generator() for g in range(nG)]
        # Set attributes of each generator object
        for key, values in data.items():
            for (g, v) in zip(generators, values):
                setattr(g, key, v)

        # Create attribute of System class
        self.generators = generators

    def add_transformers(self, data: dict) -> None:
        list_keys = list(data.keys())
        # Number of transformers
        nT = len(data[list_keys[0]])
        transformers = [Transformer() for t in range(nT)]
        # Set attributes of each transformer object
        for key, values in data.items():
            for (t, v) in zip(transformers, values):
                setattr(t, key, v)

        # Create attribute of System class
        self.transformers = transformers

    def add_conductors(self, data: dict) -> None:
        list_keys = list(data.keys())
        # Number of conductores
        nC = len(data[list_keys[0]])
        conductors = [Conductor() for c in range(nC)]
        # Set attributes of each conductor object
        for key, values in data.items():
            for (c, v) in zip(conductors, values):
                setattr(c, key, v)

        # Create attribute of System class
        self.conductors = conductors

    def set_pu(self, Sb_new: float, Vb_2: float) -> None:
        """Common base.

        It converts all impedance of the system to a common base
        where Vb_2 stands for voltage base L-L [kV]
        in region of transmission.

        """
        # Original impedance: Generatores pu
        Z0pu_old_G = [g.X0_pu for g in self.generators]
        Z1pu_old_G = [g.Xdpp_pu for g in self.generators]
        Z2pu_old_G = [g.X2_pu for g in self.generators]

        # Original impedance: Transformers pu
        Zccpu_old_T =  [t.Xcc_pu for t in self.transformers]

        # Original impedance: Conductors Ohm
        Zohm0_old_C = [c.X0_ohm for c in self.conductors]
        Zohm1_old_C = [c.X1_ohm for c in self.conductors]

        # Get original voltages generators
        Vb_old_G = [g.Vnom_kV for g in self.generators]

        # Get each transformer ratio: a
        LV_old = [p.V1nom_kV for p in self.transformers]
        HV_old = [s.V2nom_kV for s in self.transformers]
        LH_V = zip(LV_old, HV_old)
        a = [L/H for L, H in LH_V]

        # Set new voltage base in transmission region: Lines
        for L in self.conductors:
            L.Vnom_kV = Vb_2
        Vb_new_C = [v.Vnom_kV for v in self.conductors]

        # New voltages in generators
        def step_down(ratio: list) -> list:
            return ratio * Vb_2
        Vb_new_G = list(map(step_down, a))

        # Get original Sb of divices
        Sb_old_G = [g.Snom_MVA for g in self.generators]
        Sb_old_T = [t.Snom_MVA for t in self.transformers]

        def X_to_newpu(Xpu_old: list, Sb_old: list, V_old: list, V_new: list):
            return Xpu_old * ((Sb_new)/(Sb_old)) * ((V_old)/(V_new))**2

        def Xpu_lines(Xb_old, Vb_2):
            Zb = (Vb_2)**2 / (Sb_new)
            return Xb_old / Zb

        # Get new impedance in common base
        Z0pu_new_G = list(map(X_to_newpu, Z0pu_old_G, Sb_old_G, Vb_old_G, Vb_new_G))
        Z1pu_new_G = list(map(X_to_newpu, Z1pu_old_G, Sb_old_G, Vb_old_G, Vb_new_G))
        Z2pu_new_G = list(map(X_to_newpu, Z2pu_old_G, Sb_old_G, Vb_old_G, Vb_new_G))
        Zccpu_new_T = list(map(X_to_newpu, Zccpu_old_T, Sb_old_T, LV_old, LV_old))
        X0_pu_C = list(map(Xpu_lines, Zohm0_old_C, Vb_new_C))
        X1_pu_C = list(map(Xpu_lines, Zohm1_old_C, Vb_new_C))

        # Update attributes:
        for G, Xnew_pu in zip(self.generators, Z0pu_new_G):
            G.X0_pu = Xnew_pu
        for G, Xnew_pu in zip(self.generators, Z1pu_new_G):
            G.Xdpp_pu = Xnew_pu
        for G, Xnew_pu in zip(self.generators, Z2pu_new_G):
            G.X2_pu = Xnew_pu
        for T, Xnew_pu in zip(self.transformers, Zccpu_new_T):
            T.Xcc_pu = Xnew_pu
        for C, Xnew_pu in zip(self.conductors, X0_pu_C):
            C.X0_pu = Xnew_pu
        for C, Xnew_pu in zip(self.conductors, X1_pu_C):
            C.X1_pu = Xnew_pu
            C.X2_pu = Xnew_pu
        # Rest attributes remain the same.

    def store_bus(self, bus: Bus) -> None:
        """Set and creat bus.

        It create and sets in order all kind of bus.
        """
        if bus.bus_type == 'Slack':
            self.slack = bus
        elif bus.bus_type == 'PQ':
            self.PQ_buses.append(bus)
        elif bus.bus_type == 'PV':
            self.PV_buses.append(bus)
        # Define another method apart to organize it...
        # ... Call such function
        self.organize_buses()

    def organize_buses(self) -> None:

        """Organize all given buses.

        It sets buses in this order: [slack, PV, PQ].
        """
        self.non_slack_buses = self.PV_buses + self.PQ_buses
        if self.slack != None:
            self.buses = [self.slack] + self.non_slack_buses
        else:
            self.buses = self.non_slack_buses

    # Reference
    def add_slack(self, V, Vb, deg = 0, PL = 0, QL = 0, G = 0, B = 0) -> Bus:
        bus = Bus(V, deg, PL, QL, G, B, Vb, 'Slack')
        self.store_bus(bus)
        return bus

    # Load
    def add_PQ(self, B, G = 0, V = None, deg = 0, PL = None, QL = None, Vb = None) -> Bus:
        bus = Bus(V, deg, PL, QL, G, B, Vb, 'PQ')
        self.store_bus(bus)
        return bus

    # Injectors
    def add_PV(self, B, G = 0, V = None, deg = 0, PL = None, QL = None, Vb = None) -> Bus:
        bus = Bus(V, deg, PL, QL, G, B, Vb, 'PV')
        self.store_bus(bus)
        return bus

    # Model pi
    def add_line(self, from_bus, to_bus, X_pu, R_pu = 0,
                 total_G: float = 0, total_B: float = 0) -> Line:
        total_Y = total_G + 1j*total_B
        line = Line(from_bus, to_bus, R_pu, X_pu, total_Y/2, total_Y/2)
        self.lines.append(line)
        return line

    # Admitance matrix positive and negative sequence.
    def build_Y(self) -> None:
        """Admitance matrix.

        Matrix Y come to become a attribute of the ``System`` class.
        """
        N = len(self.buses)
        self.Y = np.zeros((N, N), dtype=complex)
        # Due to compensations to a single bar
        for (i, bus) in enumerate(self.buses):
            self.Y[i, i] += bus.G + 1j*bus.B
        # Due to Lines
        for line in self.lines:
            m = self.buses.index(line.from_bus)
            n = self.buses.index(line.to_bus)
            # Get series admitance of the line
            Y_serie = (1) / (line.R_pu + 1j*line.X_pu)
            # Build Y
            self.Y[m, m] += line.from_Y + Y_serie
            self.Y[n, n] += line.to_Y + Y_serie
            self.Y[m, n] -= Y_serie
            self.Y[n, m] -= Y_serie

    # Admitance matrix zero sequence.
    def build_Y0(self) -> None:
        """Admitance matrix.

        Matrix Y0 come to become a attribute of the ``System`` class.
        """

def main() -> System:
    """System data and objects.

    It gets the data of the system, based on that information
    it sets the instances as well as their attibutes and
    convert everything to a common base in p.u.
    """

    # Get data of the system
    dataSet = sysData()
    directory = './data/System_data.xlsx'
    dataSet.call_data(directory)

    # Get data
    generator_data = dataSet.generators
    tx_data = dataSet.transformers
    line_data = dataSet.lines

    # Build up a system
    sys = System()
    # Creat objects: Gegenerators, Transformers, Conductors
    sys.add_generators(generator_data)
    sys.add_transformers(tx_data)
    sys.add_conductors(line_data)
    # To common base
    sys.set_pu(1000, 765)
    return sys

def main01(sys: System) -> System:
    """Run positive sequence.

    It creates the positive sequence network of a 
    particular system.
    """

    # Get reactance
    X1_Gs = [x.Xdpp_pu for x in sys.generators]
    X1cc_Ts = [x.Xcc_pu for x in sys.transformers]
    X1_C = [x.X1_pu for x in sys.conductors]

    # To susceptance
    def to_B(X):
        return -1 / X

    B1_Gs = list(map(to_B, X1_Gs))
    # Creat buses with compensators (generators)
    b1 = sys.add_PV(B1_Gs[0])
    b2 = sys.add_PV(B1_Gs[1])
    b3 = sys.add_PV(B1_Gs[2])
    b4 = sys.add_PV(B1_Gs[3])
    b5 = sys.add_PQ(B=0)
    b6 = sys.add_PQ(B=0)
    b7 = sys.add_PQ(B=0)
    # Conductors and transformer as lines
    L1 = sys.add_line(b1, b5, X1cc_Ts[0])    # T1
    L2 = sys.add_line(b2, b6, X1cc_Ts[1])    # T2
    L3 = sys.add_line(b3, b7, X1cc_Ts[2])    # T3
    L4 = sys.add_line(b4, b7, X1cc_Ts[3])    # T4
    L56 = sys.add_line(b5, b6, X1_C[0])
    L57 = sys.add_line(b5, b7, X1_C[1])
    L67 = sys.add_line(b6, b7, X1_C[2])

    # Get admitance positive sequence:
    sys.build_Y()
    return sys

def main02(sys: System) -> System:
    """Run negative sequence.

    It creates the negative sequence network of a 
    particular system.
    """

    # Get reactance
    X2_Gs = [x.X2_pu for x in sys.generators]
    pass

def main00(sys: System) -> System:
    """Run zero sequence.

    It creates the zero sequence network of a 
    particular system.
    """

    # Get reactance
    X0_Gs = [x.X0_pu for x in sys.generators]
    X0_C = [x.X0_pu for x in sys.conductors]
    pass


if __name__ == '__main__':
    sys = main()
    sys = main01(sys)
    # Show all objects of the system:
    print('\n ** Generator **')
    for g in sys.generators:
        print(g.__dict__)
    print('\n ** Transformer **')
    for t in sys.transformers:
        print(t.__dict__)
    print('\n ** Conductor **')
    for c in sys.conductors:
        print(c.__dict__)

    # Show admitance matrix
    print(sys.Y)
