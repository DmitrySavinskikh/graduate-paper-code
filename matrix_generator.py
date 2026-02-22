import random
import os
import shutil

class MatrixGenerator():
    def __init__(
            self,
            swof_from:int,
            swof_to:int,
            sgof_from:int,
            sgof_to:int,
            pvdg_from:int,
            pvdg_to:int,
            pvto_from:int,
            pvto_to:int,
            equil_line:int,
            iters = 1,
            src_file= '',
            main_dir= '',
        ):
        self.iters = iters
        self.src_file = src_file
        self.main_dir = main_dir
        self.swof_from = swof_from
        self.swof_to = swof_to
        self.sgof_from = sgof_from
        self.sgof_to = sgof_to
        self.pvdg_from = pvdg_from
        self.pvdg_to = pvdg_to
        self.pvto_from = pvto_from
        self.pvto_to = pvto_to
        self.equil_line = equil_line

    def min2max_matrix(
            self
        ):
        output_matrix = []
        water_saturation = 0.1 # 0 - 1
        water_rel_permeab = 0.00001 # 0.00001 - 0.1
        oil_rel_permeab = 0 # 0 - 1
        water_oil_capillary = -1 # -1 - 1
        row = []
        for _ in range(15):
            row.append(round(water_saturation, 6))
            row.append(round(water_rel_permeab, 6))
            row.append(round(oil_rel_permeab, 6))
            row.append(round(water_oil_capillary, 6))
            
            # uniform distribution of parameters for matrix 4x15
            water_saturation = water_saturation + ((1 - 0.1) / 14)
            water_rel_permeab = water_rel_permeab + ((0.1 - 0.00001) / 14)
            oil_rel_permeab = oil_rel_permeab + (1 / 14)
            water_oil_capillary = water_oil_capillary + (2 / 14)

            output_matrix.append(row)
            row = []
        
        return output_matrix
    
    def random_matrix(
            self
        ):
        output_matrix = []
        row = []
        for _ in range(15):
            # uniform distribution of parameters for matrix 4x15
            water_saturation = random.uniform(0, 1)
            water_rel_permeab = random.uniform(0.00001, 0.1)
            oil_rel_permeab = random.uniform(0, 1)
            water_oil_capillary = random.uniform(-1, 1)

            row.append(round(water_saturation, 6))
            row.append(round(water_rel_permeab, 6))
            row.append(round(oil_rel_permeab, 6))
            row.append(round(water_oil_capillary, 6))

            output_matrix.append(row)
            row = []
        
        return output_matrix
    
    def create_swof_matrix(
            self
        ):
        # data logic matrix for swof

        water_saturation_step = (1 - 0.12) / (15 - 1)
        water_saturation = 0.12
        water_rel_permeab_step = 0.00001/15 + random.uniform(-2e-7, 2e-7)
        water_rel_permeab = 0
        oil_rel_permeab_step = 1/15 + random.uniform(-0.01,0.01)
        oil_rel_permeab = 1

        output_matrix = []
        row = []
        for i in range(15):
            if i == 14:
                water_rel_permeab = 0.00001
                row.append(round(water_saturation, 6))
                row.append(round(water_rel_permeab, 6))
                oil_rel_permeab = 0
                row.append(round(oil_rel_permeab, 6))
                row.append(0)
            else:
                row.append(round(water_saturation, 6))
                water_saturation += water_saturation_step
                row.append(round(water_rel_permeab, 6))
                water_rel_permeab += water_rel_permeab_step
                row.append(round(oil_rel_permeab, 6))
                oil_rel_permeab -= oil_rel_permeab_step
                row.append(0)


            # row.append(round(random.uniform(0, 0.3), 6))


            output_matrix.append(row)
            row = []
        return output_matrix
    
    def create_sgof_matrix(
            self
        ):
        # data logic matrix for sgof

        f_val = 0
        s_val = 0.001
        t_val = 0.02
        _4_val = 0.05
        gas_saturation = [f_val,s_val,t_val,_4_val,0.12,0.2,0.25,0.3,0.4,0.45,0.5,0.6,0.7,0.85,0.88]
        gas_rel_permeab = [0,0,0,0.005,0.025,0.075,0.125,0.190,0.410,0.60,0.72,0.87,0.94,0.98,0.984,1]
        oil_rel_permeab = [1,1,0.997,0.980,0.7,0.35,0.2,0.09,0.021,0.01,0.001,0.0001,0,0,0,0]

        # gas_rel_permeab_step = 0.99/15 + random.uniform(-2e-7, 2e-7)
        # gas_rel_permeab = 0
        # oil_rel_permeab_step = 1/15 + random.uniform(-0.005,0.005)
        # oil_rel_permeab = 1
        # # oil_gas_capillary = random.uniform(0, 0.15)


        output_matrix = []
        row = []
        for i in range(15):
            if i in range(0,4):
                row.append(gas_saturation[i] + random.uniform(-0.02, 0.02))
                row.append(gas_rel_permeab[i])
                row.append(oil_rel_permeab[i])
                row.append(0)
            elif i in range(10,15):
                row.append(gas_saturation[i] + random.uniform(-0.02, 0.02))
                row.append(gas_rel_permeab[i])
                row.append(oil_rel_permeab[i])
                row.append(0)
            else:
                row.append(gas_saturation[i] + random.uniform(-0.02, 0.02))
                row.append(gas_rel_permeab[i])
                row.append(oil_rel_permeab[i])
                row.append(0)
            # row.append(round(oil_gas_capillary, 3))            
            
            # oil_gas_capillary = random.uniform(-0.5, 0.5)

            output_matrix.append(row)
            row = []
        
        return output_matrix
    
    def create_pvdg_matrix(
            self
        ):
        # data logic matrix for pvdg

        f_val=14 + random.uniform(-15, 20)
        gas_phase_pres_lst = [f_val,264,514,1014,2014,2514,3014,4014,5014,9014]
        f_val=166.666 + random.uniform(-5, 20)
        s_val=12.0930 + random.uniform(-1, 10)
        gas_form_vol_factor = [f_val,s_val,6.27400,3.19700,1.61400,1.29400,1.08000,0.81100,0.64900,0.38600]
        gas_visc = [0.008000,0.009600,0.011200,0.014000,0.018900,0.020800,0.022800,0.026800,0.030900,0.047000]


        output_matrix = []
        row = []
        for i in range(10):
            row.append(gas_phase_pres_lst[i] + random.uniform(-10, 10))
            row.append(round(gas_form_vol_factor[i] + random.uniform(-0.05, 0.05), 3))
            row.append(round(gas_visc[i] + random.uniform(-0.00015, 0.00015), 4))

            output_matrix.append(row)
            row = []
        
        return output_matrix

    def create_pvto_matrix(
            self
        ):
        # data logic matrix for pvto

        dissol_gas_oil_ratio = [0.0010,0.0905,0.1800,0.3710,0.6360,0.7750,0.9300,1.2700]
        f_val=14.7 + random.uniform(-5, 20)
        s_val=264.7 + random.uniform(-40, 30)
        bubble_point_pres = [f_val,s_val,514.7,1014.7,2014.7,2514.7,3014.7,4014.7]
        oil_fvf = [1.0620,1.1500,1.2070,1.2950,1.4350,1.5,1.5650,1.6950]
        oil_visc = [1.0400,0.9750,0.9100,0.8300,0.6950,0.6410,0.5940,0.5100]


        output_matrix = []
        row = []
        for i in range(8):
            row.append(round(dissol_gas_oil_ratio[i] + random.uniform(-0.03, 0.03), 4))
            if i > 3:
                row.append(bubble_point_pres[i] + random.uniform(-75, 75))
            else:
                row.append(bubble_point_pres[i])
            row.append(round(oil_fvf[i] + random.uniform(-0.002, 0.002), 4))
            row.append(round(oil_visc[i] + random.uniform(-0.015, 0.015), 4))

            output_matrix.append(row)
            row = []
        
        return output_matrix
    
    def create_equil_matrix(
            self
        ):
        # data logic matrix for equil
        
        return [[8400 + random.uniform(-10, 10), 4800 + random.uniform(-300, 300), 8450 + random.uniform(-10, 10), 0, 8300 + random.uniform(-50, 50), random.randint(0,1), 1, 0, 0]]

    def format_matrix(
            self,
            matrix
        ):
        output = []
        for row in matrix:
            formatted_row = '\t'.join([
                f"{int(value):d}" if value.is_integer() else f"{value:.6f}" 
                for value in row
            ]) # ai
            output.append(formatted_row)
        output[-1] = output[-1] + ' /'
        
        return output
    
    def format_pvto_matrix(
            self,
            matrix
        ):
        output = []
        for row in matrix:
            formatted_row = '\t'.join([f"{value:.6f}" for value in row]) # ai / it's for match with format in orig file
            formatted_row += ' /'
            output.append(formatted_row)
        
        output[-1] = output[-1][:-2]
        return output

    def insert_matrix(
            self,
            file: str,
            input_block: str,
            from_: int,
            to_: int
        ):
            with open(file, 'r') as f:
                lines = f.readlines()

            lines[from_-1:to_ + 1] = [line + '\n' for line in input_block]

            with open(file, 'w') as f:
                f.writelines(lines)


    def make_files(
            self
        ):

        for i in range(self.iters):
            output_file = os.path.join(self.main_dir, f'SPE1CASE1_ITER_{i}.DATA')
            shutil.copy2(self.src_file, output_file)
            
            swof_matrix = self.create_swof_matrix()
            formatted_swof_matrix = self.format_matrix(swof_matrix)
            self.insert_matrix(output_file, formatted_swof_matrix, self.swof_from, self.swof_to)

            # sgof_matrix = self.create_sgof_matrix()
            # formatted_sgof_matrix = self.format_matrix(sgof_matrix)
            # self.insert_matrix(output_file, formatted_sgof_matrix, self.sgof_from, self.sgof_to)

            pvdg_matrix = self.create_pvdg_matrix()
            formatted_pvdg_matrix = self.format_matrix(pvdg_matrix)
            self.insert_matrix(output_file, formatted_pvdg_matrix, self.pvdg_from, self.pvdg_to)

            pvto_matrix = self.create_pvto_matrix()
            formatted_pvto_matrix = self.format_pvto_matrix(pvto_matrix)
            self.insert_matrix(output_file, formatted_pvto_matrix, self.pvto_from, self.pvto_to)

            equil_matrix = self.create_equil_matrix()
            formatted_equil_matrix = self.format_matrix(equil_matrix)
            self.insert_matrix(output_file, formatted_equil_matrix, self.equil_line, self.equil_line)


if __name__ == '__main__':
    generator = MatrixGenerator(
        iters=2, 
        src_file='/home/dmitrysavinskikh/data/opm-data/spe1/SPE1CASE1.DATA', 
        main_dir='/home/dmitrysavinskikh/graduate-paper-code/spe1',
        swof_from=144,
        swof_to=157,
        sgof_from=169,
        sgof_to=182,
        pvdg_from=211,
        pvdg_to=219,
        pvto_from=229,
        pvto_to=235,
        equil_line=273
    )
    generator.make_files()

    