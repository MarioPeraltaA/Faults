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
import numpy as np


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


class Bus:

    def __init__(self,
                 V: float,
                 deg: float,
                 PL: float,
                 QL: float,
                 G: float,
                 B: float,
                 Vb: float,
                 bus_type: str,
                 aux: bool = False):
        self.V = V
        self.theta = deg*np.pi / 180      # To rad
        self.PL = PL
        self.QL = QL
        self.G = G     # Conductance of compensation
        self.B = B     # Susceptance of compensation
        self.Vb = Vb
        self.bus_type = bus_type
        self.aux = aux

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
                 to_Y: complex,
                 Tx: bool = False):
        self.from_bus = from_bus
        self.to_bus = to_bus
        self.R_pu = R_pu
        self.X_pu = X_pu
        self.from_Y = from_Y   # Half shunt
        self.to_Y = to_Y       # Other half shunt
        self.Tx = Tx           # See as transformer


class System:
    Sb = 100    # MVA
    Vb = 138    # kV
    r"""Network.

    ``System`` class represents the electric network.

    It makes up an auxiliary list in order to organize the buses.
    It receives seq_type being as the kind of sequence: 0 for zero,
    1 for positive and 2 for negative.
    """
    def __init__(self, seq_type: int = 1):
        self.slack = None
        self.buses = []
        self.non_slack_buses = []
        self.PQ_buses = []
        self.PV_buses = []
        self.lines = []
        self.Y_012 = [None, None, None]
        self.seq_type = seq_type

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

    def set_pu(self, Sb_new: float, Vb_new: float) -> None:
        """Common base.

        It converts and updates all impedance of the system
        to a common base while other attributes remain the same,
        where Sb_new stands for new global apparent power base
        in [MVA] and Vb_new the voltage base L-L in [kV] in region
        of transmission.
        """
        # Set arguments as new globla class variables
        System.Sb = Sb_new
        System.Vb = Vb_new
        # Original impedance: Generatores pu
        Z0pu_old_G = [g.X0_pu for g in self.generators]
        Z1pu_old_G = [g.Xdpp_pu for g in self.generators]
        Z2pu_old_G = [g.X2_pu for g in self.generators]
        # Due to connection.
        Xgpu_old_G = [g.Xg_pu for g in self.generators]

        # Original impedance: Transformers pu
        Zccpu_old_T = [t.Xcc_pu for t in self.transformers]

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

        # Update attribute to new voltage base in transmission region: Lines
        for L in self.conductors:
            L.Vnom_kV = Vb_new
        Vb_new_C = [v.Vnom_kV for v in self.conductors]

        # New voltages in generators
        def step_down(ratio: list) -> list:
            return ratio * Vb_new
        Vb_new_G = list(map(step_down, a))
        # Update attribute
        for (g, v) in zip(self.generators, Vb_new_G):
            g.Vnom_kV = v

        # Get original Sb of divices
        Sb_old_G = [g.Snom_MVA for g in self.generators]
        Sb_old_T = [t.Snom_MVA for t in self.transformers]

        def X_to_newpu(Xpu_old: list, Sb_old: list, V_old: list, V_new: list):
            return Xpu_old * ((Sb_new)/(Sb_old)) * ((V_old)/(V_new))**2

        def Xpu_lines(Xb_old, Vb_2):
            Zb = (Vb_2)**2 / (Sb_new)
            return Xb_old / Zb

        # Get new impedance in common base
        Z0pu_new_G = list(map(X_to_newpu,
                              Z0pu_old_G,
                              Sb_old_G,
                              Vb_old_G,
                              Vb_new_G))
        Z1pu_new_G = list(map(X_to_newpu,
                              Z1pu_old_G,
                              Sb_old_G,
                              Vb_old_G,
                              Vb_new_G))
        Z2pu_new_G = list(map(X_to_newpu,
                              Z2pu_old_G,
                              Sb_old_G,
                              Vb_old_G,
                              Vb_new_G))
        Xgpu_old_G = list(map(X_to_newpu,
                              Xgpu_old_G,
                              Sb_old_G,
                              Vb_old_G,
                              Vb_new_G))
        Zccpu_new_T = list(map(X_to_newpu,
                               Zccpu_old_T,
                               Sb_old_T,
                               LV_old,
                               LV_old))
        X0_pu_C = list(map(Xpu_lines, Zohm0_old_C, Vb_new_C))
        X1_pu_C = list(map(Xpu_lines, Zohm1_old_C, Vb_new_C))

        # Update attributes:
        for G, Xnew_pu in zip(self.generators, Z0pu_new_G):
            G.X0_pu = Xnew_pu
        for G, Xnew_pu in zip(self.generators, Z1pu_new_G):
            G.Xdpp_pu = Xnew_pu
        for G, Xnew_pu in zip(self.generators, Z2pu_new_G):
            G.X2_pu = Xnew_pu
        for G, Xgnew_pu in zip(self.generators, Xgpu_old_G):
            G.Xg_pu = Xgnew_pu
        for T, Xnew_pu in zip(self.transformers, Zccpu_new_T):
            T.Xcc_pu = Xnew_pu
        for C, Xnew_pu in zip(self.conductors, X0_pu_C):
            C.X0_pu = Xnew_pu
        for C, Xnew_pu in zip(self.conductors, X1_pu_C):
            C.X1_pu = Xnew_pu
            C.X2_pu = Xnew_pu

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
        """Organize given bus.

        It sets buses in this order: [slack, PV, PQ].
        """
        self.non_slack_buses = self.PV_buses + self.PQ_buses
        if self.slack is not None:
            self.buses = [self.slack] + self.non_slack_buses
        else:
            self.buses = self.non_slack_buses

    # Reference
    def add_slack(self, V, Vb, deg=0, PL=0, QL=0, G=0, B=0) -> Bus:
        bus = Bus(V, deg, PL, QL, G, B, Vb, 'Slack')
        self.store_bus(bus)
        return bus

    # Load
    def add_PQ(self, B, Vb, G=0, V=None, deg=0, PL=None, QL=None, aux=False) -> Bus:
        bus = Bus(V, deg, PL, QL, G, B, Vb, 'PQ', aux)
        self.store_bus(bus)
        return bus

    # Injectors
    def add_PV(self, B, Vb, G=0, V=None, deg=0, PL=None, QL=None, aux=False) -> Bus:
        bus = Bus(V, deg, PL, QL, G, B, Vb, 'PV', aux)
        self.store_bus(bus)
        return bus

    # Model pi
    def add_line(self, from_bus, to_bus, X_pu, R_pu=0,
                 total_G: float = 0,
                 total_B: float = 0,
                 Tx: bool = False) -> Line:
        total_Y = total_G + 1j*total_B
        line = Line(from_bus, to_bus, R_pu, X_pu, total_Y/2, total_Y/2, Tx)
        self.lines.append(line)
        return line

    def build_Y(self) -> None:
        """Admitance matrix of any sequence network.

        Matrix Y come to become a attribute of the ``System`` class.
        """
        if self.seq_type == 0:
            # Transformers
            Lx = [t for t in self.lines if t.Tx]

            for L, T in zip(Lx, self.transformers):
                b0_from = self.add_PQ(-1/1e6, T.V1nom_kV, aux=True)    # Aux. load bus from
                b0_to = self.add_PQ(-1/1e6, T.V2nom_kV, aux=True)      # Aux. load bus to
                b_to = L.to_bus  # Save old toward
                self.add_line(b0_from, b0_to, T.Xcc_pu)
                # Update toward
                L.to_bus = b0_from
                # Primary
                if T.Conn1 == 'D':
                    L.X_pu = 1e6
                    b0_from.B = -1/1e-6
                elif T.Conn1 == 'Yg':
                    L.X_pu = 1e-6
                elif T.Conn1 == 'Yn':
                    # No data
                    L.X_pu = 3 * 0.05
                elif T.Conn1 == 'Y':
                    L.X_pu = 1e6

                # Secondary
                if T.Conn2 == 'D':
                    self.add_line(b0_to, b_to, 1e6)
                    b0_to.B = -1/1e-6
                elif T.Conn2 == 'Yg':
                    self.add_line(b0_to, b_to, 1e-6)
                elif T.Conn2 == 'Yn':
                    # No data
                    self.add_line(b0_to, b_to, 3*0.5)
                elif T.Conn2 == 'Y':
                    self.add_line(b0_to, b_to, 1e6)

            # Generators
            for b, g in zip(self.PV_buses, self.generators):
                if g.Conn == 'Yg':
                    b.B += 1e-6
                elif g.Conn == 'Yn':
                    b.B += -1 / (3*g.Xg_pu)
                elif g.Conn == 'Y':
                    b.B = -1/1e6

            # Creat empty Y zero sequence matrix
            N = len(self.buses)
            Y0 = np.zeros((N, N), dtype=complex)
            # Due to compensations to a single bar
            for (i, bus) in enumerate(self.buses):
                Y0[i, i] += bus.G + 1j*bus.B
            # Due to Lines
            for line in self.lines:
                m = self.buses.index(line.from_bus)
                n = self.buses.index(line.to_bus)
                # Get series admitance of the line
                Y_serie = (1) / (line.R_pu + 1j*line.X_pu)
                # Build Y
                Y0[m, m] += line.from_Y + Y_serie
                Y0[n, n] += line.to_Y + Y_serie
                Y0[m, n] -= Y_serie
                Y0[n, m] -= Y_serie
            # Reduce Y
            nB = len([b for b in self.buses if not b.aux])
            K = Y0[:nB, :nB]
            L = Y0[:nB, nB:]
            N = Y0[nB:, :nB]
            M = Y0[nB:, nB:]
            Y0 = K - np.matmul(np.matmul(L, np.linalg.inv(M)), N)
            self.Y_012[0] = Y0

        else:
            N = len(self.buses)
            Y12 = np.zeros((N, N), dtype=complex)
            # Due to compensations to a single bar
            for (i, bus) in enumerate(self.buses):
                Y12[i, i] += bus.G + 1j*bus.B
            # Due to Lines
            for line in self.lines:
                m = self.buses.index(line.from_bus)
                n = self.buses.index(line.to_bus)
                # Get series admitance of the line
                Y_serie = (1) / (line.R_pu + 1j*line.X_pu)
                # Build Y
                Y12[m, m] += line.from_Y + Y_serie
                Y12[n, n] += line.to_Y + Y_serie
                Y12[m, n] -= Y_serie
                Y12[n, m] -= Y_serie
            self.Y_012[self.seq_type] = Y12

    @property
    def Z_012(self) -> tuple[np.ndarray]:
        """Impedance Matrix.

        It returns the tuple of impedance matrix of each sequence.
        """
        Y0 = self.Y_012[0]
        Y1 = self.Y_012[1]
        Y2 = self.Y_012[2]
        Z0 = np.linalg.inv(Y0)
        Z1 = np.linalg.inv(Y1)
        Z2 = np.linalg.inv(Y2)
        return (Z0, Z1, Z2)

    def to_phase(self, S_012: np.ndarray) -> np.ndarray:
        """Matrix A.

        In order to switch from sequence domain to phase domain.
        """
        a = -1/2 + np.sqrt(3)*1j/2
        A = np.array([[1, 1, 1], [1, a**2, a], [1, a, a**2]])
        return A @ S_012

    def get_allVoltages(self,
                        Bf: Bus,
                        Vf: complex,
                        Z_012: np.ndarray[complex],
                        I_012: np.ndarray[complex]) -> np.ndarray[complex]:
        I_0, I_1, I_2 = I_012
        Z_0, Z_1, Z_2 = Z_012
        b = self.buses.index(Bf)       # Index of faulted bus
        nB = len(self.buses)           # Number of buses
        # Current is zero in all buses but the
        # faulted bus and such leaves the system, then:
        Isys0 = np.zeros(nB, dtype=complex)
        Isys1 = np.zeros(nB, dtype=complex)
        Isys2 = np.zeros(nB, dtype=complex)
        # Update current at faulted bus
        Isys0[b] = -I_0
        Isys1[b] = -I_1
        Isys2[b] = -I_2
        # Pre fault Vf source column vector
        VF = np.full(nB, Vf, dtype=complex)
        V0_ckt = Z_0 @ Isys0        # Zero
        V1_ckt = VF + Z_1@Isys1     # Positive
        V2_ckt = Z_2 @ Isys2        # Negative
        # Stask horizontally as columns
        V012_ckt = np.column_stack((V0_ckt, V1_ckt, V2_ckt))
        return V012_ckt

    def balanced(self,
                 B: int,
                 Vf: complex = 1,
                 Zf: float = 1e-6) -> dict[list[np.ndarray[complex]]]:
        """Three-phase-to-ground balanced fault.

        Sets new attribute that contain both current and voltages
        fault in pu of each phase as dictionary whose key is the faulted bus.
        It calculates the balanced current fault and voltages in phase domain
        of the circuit during fault in p.u. of a arcing fault
        at arbitrary bus ``B`` through Zf [Ohm] as general case.
        If Zf is small then it is a bolted short circuit kind of fault.

        Note: It only takes positive sequence of arbitrary phase.
        """
        b = B - 1               # Index of faulted bus
        Bf = self.buses[b]      # Faulted bus instance
        Vbase = Bf.Vb           # Base voltage
        # Base impedance at region of bus B
        Zb = (Vbase)**2 / (System.Sb)
        Zf_pu = Zf / Zb         # To p.u.

        Z012 = self.Z_012       # Impedance matrix
        Z1 = Z012[1]            # Positive seq. only
        Z_TH = Z1[b, b]         # Z_TH from bus B

        # Current sequence arcing fault in pu
        I1f_pu = (Vf) / (Z_TH + Zf_pu)        # Positive seq.
        I012 = np.array([0, I1f_pu, 0])       # Column vector of I sequences

        # Voltages during fault in pu
        V012ckt = self.get_allVoltages(Bf, Vf, Z012, I012)
        # To phase domain
        Iabc_pu = self.to_phase(I012)
        nB = len(self.buses)
        Vabc_pu = []
        for i in range(nB):
            Vabc_pu.append(self.to_phase(V012ckt[i]))

        # Set new attribute
        self.balanced_fault = {
            Bf: [Iabc_pu, Vabc_pu]
        }
        return Bf

    def single(self,
              B: int,
              Vf: complex = 1,
              Zf: float = 1e-6) -> dict[list[np.ndarray[complex]]]:
        """Single line-to-ground fault.

        Sets new attribute that contain both current and voltages
        fault in pu of each phase.
        Consider a single line-to-ground fault from arbitrary phase to ground
        at the general three-phase bus. For generality, a fault
        impedance Zf is included. In the case of a bolted
        fault, Zf = 1e-6 [Ohm], whereas for an arcing fault,
        Zf is the arc impedance.
        """
        b = B - 1               # Index of faulted bus
        Bf = self.buses[b]      # Faulted bus instance
        Vbase = Bf.Vb           # Base voltage
        # Base impedance at region of bus B
        Zb = (Vbase)**2 / (System.Sb)
        Zf_pu = Zf / Zb         # To p.u.

        Z012 = self.Z_012       # Impedance matrix
        Z0 = Z012[0]
        Z0_TH = Z0[b, b]
        Z1 = Z012[1]
        Z1_TH = Z1[b, b]
        Z2 = Z012[2]
        Z2_TH = Z2[b, b]

        # Current sequence arcing fault in pu
        I0f_pu = (Vf) / (Z0_TH+Z1_TH+Z2_TH+3*Zf_pu)    # Zero seq.
        I1f_pu = I0f_pu              # Positive seq.
        I2f_pu = I0f_pu              # Negative seq.
        I012 = np.array([I0f_pu, I1f_pu, I2f_pu])     # Column vector of I sequences

        # Voltages during fault in pu
        V012ckt = self.get_allVoltages(Bf, Vf, Z012, I012)
        # To phase domain
        Iabc_pu = self.to_phase(I012)   # Current at fault location
        Vabc_pu = []                    # All voltages of the network
        nB = len(self.buses)
        for i in range(nB):
            Vabc_pu.append(self.to_phase(V012ckt[i]))

        # Set new attribute
        self.single_fault = {
            Bf: [Iabc_pu, Vabc_pu]
        }
        return Bf

    def line_to_line(self,
                     B: int,
                     Vf: complex = 1,
                     Zf: float = 1e-6) -> dict[list[np.ndarray[complex]]]:
        """Line-to-line fault.

        Sets new attribute that contain both current and voltages
        fault in pu  of each phase.
        Consider a line-to-line fault between two arbitrary phases.
        It includes a fault impedance ZF for generality.
        """
        b = B - 1               # Index of faulted bus
        Bf = self.buses[b]      # Faulted bus instance
        Vbase = Bf.Vb           # Base voltage
        # Base impedance at region of bus B
        Zb = (Vbase)**2 / (System.Sb)
        Zf_pu = Zf / Zb         # To p.u.

        Z012 = self.Z_012       # Impedance matrix
        Z1 = Z012[1]
        Z1_TH = Z1[b, b]
        Z2 = Z012[2]
        Z2_TH = Z2[b, b]

        # Current sequence arcing fault in pu
        I0f_pu = 0    # Zero seq.
        I1f_pu = Vf / (Z1_TH+Z2_TH+Zf_pu)    # Positive seq.
        I2f_pu = -I1f_pu                      # Negative seq.
        I012 = np.array([I0f_pu, I1f_pu, I2f_pu])     # Column vector of I sequences

        # Voltages during fault in pu
        V012ckt = self.get_allVoltages(Bf, Vf, Z012, I012)
        # To phase domain
        Iabc_pu = self.to_phase(I012)   # Current at fault location
        Vabc_pu = []                    # All voltages of the network
        nB = len(self.buses)
        for i in range(nB):
            Vabc_pu.append(self.to_phase(V012ckt[i]))

        # Set new attribute
        self.line_line_fault = {
            Bf: [Iabc_pu, Vabc_pu]
        }
        return Bf

    def double_to_ground(self,
               B: int,
               Vf: complex = 1,
               Zf: float = 1e-6) -> tuple[np.ndarray[complex]]:
        """Double line-to-ground fault.

        Sets new attribute that contain both current and voltages
        fault in pu of each phase.
        A double line-to-ground fault between two arbitrary phases
        through fault impedance Zf to ground.
        In the case of a bolted fault, Zf = 1e-6 [Ohm],
        whereas for an arcing fault, Zf is the arc impedance.
        """
        b = B - 1               # Index of faulted bus
        Bf = self.buses[b]      # Faulted bus instance
        Vbase = Bf.Vb           # Base voltage
        # Base impedance at region of bus B
        Zb = (Vbase)**2 / (System.Sb)
        Zf_pu = Zf / Zb         # To p.u.

        Z012 = self.Z_012       # Impedance matrix
        Z0 = Z012[0]
        Z0_TH = Z0[b, b]
        Z1 = Z012[1]
        Z1_TH = Z1[b, b]
        Z2 = Z012[2]
        Z2_TH = Z2[b, b]

        # Current sequence arcing fault in pu
        Zeq = (Z2_TH*(Z0_TH+3*Zf_pu)) / (Z0_TH+Z2_TH+3*Zf_pu)
        I1f_pu = (Vf) / (Z1_TH+Zeq)                        # Positive seq.
        I0f_pu = (-I1f_pu*Z2_TH) / (Z0_TH+Z2_TH+3*Zf_pu)   # Zero seq.
        I2f_pu = (-I1f_pu*(Z0_TH+3*Zf_pu)) / (Z0_TH+Z2_TH+3*Zf_pu)    # Negative seq.
        I012 = np.array([I0f_pu, I1f_pu, I2f_pu])     # Column vector of I sequences

        # Voltages during fault in pu
        V012ckt = self.get_allVoltages(Bf, Vf, Z012, I012)
        # To phase domain
        Iabc_pu = self.to_phase(I012)   # Current at fault location
        Vabc_pu = []                    # All voltages of the network
        nB = len(self.buses)
        for i in range(nB):
            Vabc_pu.append(self.to_phase(V012ckt[i]))

        # Set new attribute
        self.double_fault = {
            Bf: [Iabc_pu, Vabc_pu]
        }
        return Bf

def main() -> System:
    """System data and objects.

    It gets the data of the system, based on the information.
    It sets the instances as well as their attibutes and
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


def sys_012(sys: System) -> System:

    # To susceptance
    def to_B(X):
        return -1 / X

    # Get new voltages base of each generators
    Vb_G = [g.Vnom_kV for g in sys.generators]

    # Get zero seq. reactance
    X0_Gs = [x.X0_pu for x in sys.generators]
    X0cc_Ts = [x.Xcc_pu for x in sys.transformers]
    X0_C = [x.X0_pu for x in sys.conductors]          
    # Get positive seq. reactance
    X1_Gs = [x.Xdpp_pu for x in sys.generators]
    X1cc_Ts = [x.Xcc_pu for x in sys.transformers]
    X1_C = [x.X1_pu for x in sys.conductors]
    # Get negative seq. reactance
    X2_Gs = [x.X2_pu for x in sys.generators]
    X2cc_Ts = [x.Xcc_pu for x in sys.transformers]
    X2_C = [x.X2_pu for x in sys.conductors]
    # Generators as compensators, set X as susceptance
    B0_Gs = list(map(to_B, X0_Gs))
    B1_Gs = list(map(to_B, X1_Gs))
    B2_Gs = list(map(to_B, X2_Gs))
    # Store in iterable
    seq = {
        0: [B0_Gs, X0cc_Ts, X0_C],
        1: [B1_Gs, X1cc_Ts, X1_C],
        2: [B2_Gs, X2cc_Ts, X2_C]
    }

    # Get admitance matrix for each sequence
    for k, values in seq.items():
        # Initialize
        sys.buses = []
        sys.non_slack_buses = []
        sys.PQ_buses = []
        sys.PV_buses = []
        sys.lines = []
        sys.seq_type = k

        for n, p in enumerate(values):
            # Creat buses with compensators (generators)
            if n == 0:
                b1 = sys.add_PV(p[0], Vb_G[0])
                b2 = sys.add_PV(p[1], Vb_G[1])
                b3 = sys.add_PV(p[2], Vb_G[2])
                b4 = sys.add_PV(p[3], Vb_G[3])
                # Load buses
                b5 = sys.add_PQ(0, System.Vb)
                b6 = sys.add_PQ(0, System.Vb)
                b7 = sys.add_PQ(0, System.Vb)
                continue
            # Transformers
            elif n == 1:
                T1 = sys.add_line(b1, b5, p[0], Tx=True)
                T2 = sys.add_line(b2, b6, p[1], Tx=True)
                T3 = sys.add_line(b3, b7, p[2], Tx=True)
                T4 = sys.add_line(b4, b7, p[3], Tx=True)
                continue
            # Lines
            elif n == 2:
                L56 = sys.add_line(b5, b6, p[0])
                L57 = sys.add_line(b5, b7, p[1])
                L67 = sys.add_line(b6, b7, p[2])
        # Get Y admitance per sequence
        sys.build_Y()
    return sys


if __name__ == '__main__':
    sys = main()                 # Get system
    sys012 = sys_012(sys=sys)    # Y matrix of each sequence
