"""Sequences of a transformer.

This program determines the three-sequence networks of a transformer
either delta or why with two or three windings. The user specifies
the kind of connection of each winding: Delta or Wye
(floating, bolted or through a impedance). For sequence zero
is going to return a list of a combination of edges and nodes
of the following network:

Switch method.

        (1)*--o o--(2)-----(5)-----(3)---o o---*(4)
                    |       |       |
                    |      (6)-----------o o---*(7)
                    |       |       |
                    o       o       o
                    o       o       o
                    |       |       |
         *-------------------------------------*(0).

In case Wye is floating edge would not be considered.

Mario R. Peralta A.
Electric Power and Energy Research Laboratory (EPERLab).
Escuela de Ingeniería Eléctrica (EIE).

"""

class Delta:

    def __init__(self, position: int):
        s = {1, 2, 3}
        if position in s:
            self.position = position
            self.kind = 'Delta'
        else:
            print('WindingPositionError: winding can only have 1, 2, 3 position.')
            import sys
            sys.exit()
class Wye:

    def __init__(self, position: int, GRN: bool = True, Yg: complex = None):
        s = {1, 2, 3}
        if position in s:
            self.position = position
        else:
            print('WindingPositionError: winding can only have 1, 2, 3 position.')
            import sys
            sys.exit()
        self.GRN = GRN
        self.Yg = Yg
        if GRN and not Yg:
            self.kind = 'Bolted'         # S.C.
        elif GRN:
            self.kind = 'Ground'        # Through impedance
        elif not GRN:
            self.Yg = None
            self.kind = 'Floating'      # O.C.

class Transformer:

    def __init__(self, *args):
        self.edges = []
        self.nodes = []
        places = {p.position for p in args}
        P = len(places)
        if P != len(args):
            print('WindingPositionError: 2 or more windings cannot be in the same position.')
            import sys
            sys.exit()
        if max(places) != len(places):
            print('WindingMissinError.')
            import sys
            sys.exit()

        self.windings = args


    def seq0(self):
        """Switch method.

        (1)*--o o--(2)-----(5)-----(3)---o o---*(4)
                    |       |       |
                    |      (6)-----------o o---*(7)
                    |       |       |
                    o       o       o
                    o       o       o
                    |       |       |
         *-------------------------------------*(0).

        """
        vertices = ['0', '1', '2', '3', '4', '5', '6', '7']
        branches = [{'2', '5'}, {'5', '3'}, {'5', '6'}]
        N = len(self.windings)
        if N == 2:
            # Minimun nodes and edges
            branches.remove(branches[-1])
            for v in vertices[:-2]:
                self.nodes.append(v)
            # Possible nodes and edges
            for w in self.windings:
                if w.position == 1:
                    if type(w) is Delta:
                        branches.append({'0', '2'})
                    else:
                        if w.kind == 'Bolted':
                            branches.append({'1', '2'})
                        elif w.kind == 'Ground':
                            branches.append(['1==', '==2'])
                else:
                    if type(w) is Delta:
                        branches.append({'0', '3'})
                    else:
                        if w.kind == 'Bolted':
                            branches.append({'3', '4'})
                        elif w.kind == 'Ground':
                            branches.append(['3==', '==4'])
            self.edges = branches
            return self.edges
        else:
            self.nodes = vertices
            for w in self.windings:
                if w.position == 1:
                    if type(w) is Delta:
                        branches.append({'0', '2'})
                    else:
                        if w.kind == 'Bolted':
                            branches.append({'1', '2'})
                        elif w.kind == 'Ground':
                            branches.append(['1==', '==2'])

                elif w.position == 2:
                    if type(w) is Delta:
                        branches.append({'0', '3'})
                    else:
                        if w.kind == 'Bolted':
                            branches.append({'3', '4'})
                        elif w.kind == 'Ground':
                            branches.append(['3==', '==4'])

                else:
                    if type(w) is Delta:
                        branches.append({'0', '6'})
                    else:
                        if w.kind == 'Bolted':
                            branches.append({'6', '7'})
                        elif w.kind == 'Ground':
                            branches.append(['6==', '==7'])

            self.edges = branches
            return self.edges

    def seq1(self):
        """Positive sequence.

        (1)*----------(0)---------*(2)
                       | 
                       |
                       |----------*(3)

        """
        # Two windings
        N = len(self.windings)
        vertices = ['0', '1', '2', '3']
        branches = [{'0', '1'}, {'0', '2'}, {'0', '3'}]
        if N == 2:
            self.nodes = vertices[:-1]
            self.edges = branches[:-1]
            return self.edges
        # Three windings
        else:
            self.nodes = vertices
            self.edges = branches
            return self.edges

    def seq2(self):
        """Negative sequence.

        Same as positive sequence.
        """
        return self.seq1()
