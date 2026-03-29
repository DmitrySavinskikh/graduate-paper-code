import random
import os
import shutil
import pandas as pd

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
            self,
            N_w: int,
            N_o: int
        ):
        # data logic matrix for swof

        output_df = pd.DataFrame(columns=['water_saturation', 'water_relative_permeability', 'oil_relative_permeability', 'water_oil_capillary_pressure'])

        water_saturation_step = (1 - 0.035) / (15 - 1)
        water_saturation = 0.035
        output_matrix = []
        row = []
        for i in range(15):
            water_rel_permeab = (water_saturation ** N_w ) * (10**-5)
            oil_rel_permeab = (1 - water_saturation) ** N_o
            if i == 14:
                row.append(round(water_saturation, 6))
                row.append(round(water_rel_permeab, 6))
                row.append(round(oil_rel_permeab, 6))
                row.append(0)
            else:
                row.append(round(water_saturation, 6))
                water_saturation += water_saturation_step
                row.append(round(water_rel_permeab, 6))
                row.append(round(oil_rel_permeab, 6))
                row.append(0)

            output_matrix.append(row)
            output_df.loc[len(output_df)] = row
            row = []
        return output_matrix, output_df
    
    def create_sgof_matrix(
            self
        ):
        # data logic matrix for sgof

        output_df = pd.DataFrame(columns=['gas_saturation', 'gas_relative_permeability', 'oil_relative_permeability', 'oil_gas_capillary_pressure'])

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
                row.append(gas_saturation[i] + random.uniform(-0.04, 0.04))
                row.append(gas_rel_permeab[i])
                row.append(oil_rel_permeab[i])
                row.append(0)
            elif i in range(10,15):
                row.append(gas_saturation[i] + random.uniform(-0.04, 0.04))
                row.append(gas_rel_permeab[i])
                row.append(oil_rel_permeab[i])
                row.append(0)
            else:
                row.append(gas_saturation[i] + random.uniform(-0.04, 0.04))
                row.append(gas_rel_permeab[i])
                row.append(oil_rel_permeab[i])
                row.append(0)
            # row.append(round(oil_gas_capillary, 3))            
            
            # oil_gas_capillary = random.uniform(-0.5, 0.5)

            output_df.loc[len(output_df)] = row
            output_matrix.append(row)
            row = []
        
        return output_matrix, output_df
    
    def create_pvdg_matrix(
            self
        ):
        # data logic matrix for pvdg

        output_df = pd.DataFrame(columns=['gas_phase_pressure', 'gas_formation_volume_factor', 'gas_viscosity'])

        f_val=14 + random.uniform(-15, 20)
        gas_phase_pres_lst = [f_val,264,514,1014,2014,2514,3014,4014,5014,9014]
        f_val=166.666 + random.uniform(-5, 20)
        s_val=12.0930 + random.uniform(-1, 10)
        gas_form_vol_factor = [f_val,s_val,6.27400,3.19700,1.61400,1.29400,1.08000,0.81100,0.64900,0.38600]
        gas_visc = [0.008000,0.009600,0.011200,0.014000,0.018900,0.020800,0.022800,0.026800,0.030900,0.047000]


        output_matrix = []
        row = []
        for i in range(10):
            row.append(gas_phase_pres_lst[i] + random.uniform(-20, 20))
            row.append(round(gas_form_vol_factor[i] + random.uniform(-0.08, 0.08), 3))
            row.append(round(gas_visc[i] + random.uniform(-0.0003, 0.0003), 4))
            
            output_df.loc[len(output_df)] = row
            output_matrix.append(row)
            row = []
        
        return output_matrix, output_df

    def create_pvto_matrix(
            self
        ):
        # data logic matrix for pvto
        output_df = pd.DataFrame(columns=['dissolved_gas_oil_ratio', 'bubble_point_pressure', 'oil_fvf_for_saturated_oil', 'oil_viscosity_for_saturated_oil'])

        dissol_gas_oil_ratio = [0.0010,0.0905,0.1800,0.3710,0.6360,0.7750,0.9300,1.2700]
        f_val=14.7 + random.uniform(-10, 40)
        s_val=264.7 + random.uniform(-40, 30)
        bubble_point_pres = [f_val,s_val,514.7,1014.7,2014.7,2514.7,3014.7,4014.7]
        oil_fvf = [1.0620,1.1500,1.2070,1.2950,1.4350,1.5,1.5650,1.6950]
        oil_visc = [1.0400,0.9750,0.9100,0.8300,0.6950,0.6410,0.5940,0.5100]


        output_matrix = []
        row = []
        for i in range(8):
            row.append(round(dissol_gas_oil_ratio[i] + random.uniform(-0.06, 0.06), 4))
            if i > 3:
                row.append(bubble_point_pres[i] + random.uniform(-150, 150))
            else:
                row.append(bubble_point_pres[i])
            row.append(round(oil_fvf[i] + random.uniform(-0.004, 0.004), 4))
            row.append(round(oil_visc[i] + random.uniform(-0.03, 0.03), 4))

            output_df.loc[len(output_df)] = row
            output_matrix.append(row)
            row = []
        
        return output_matrix, output_df
    
    def create_equil_matrix(
            self
        ):
        # data logic matrix for equil
        output_df = pd.DataFrame(columns=['datum_depth', 'pressure_at_datum_depth', 'depth_of_water_oil_contact', 'oil_water_capillary_pressure_at_water_oil_contact', 'depth_of_gas_oil_contact', 'gas_oil_capillary_pressure_at_gas_oil_contact', 'rsvd_table', 'rvvd_table', 'set_to_0'])
        output_lst = [[8400 + random.uniform(-15, 15), 4800 + random.uniform(-300, 300), 8450 + random.uniform(-15, 15), 0, 8300 + random.uniform(-60, 60), random.randint(0,1), 1, 0, 0]]
        output_df.loc[len(output_df)] = output_lst[0]
        return output_lst, output_df

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


    # def make_files(
    #         self
    #     ):
    #     final_df = pd.DataFrame()

    #     for i in range(self.iters):
    #         output_file = os.path.join(self.main_dir, f'SPE1CASE1_ITER_{i}.DATA')
    #         shutil.copy2(self.src_file, output_file)
            
    #         swof_matrix, swof_df = self.create_swof_matrix(N_w=random.randint(1, 5), N_o=random.randint(1, 5))
    #         formatted_swof_matrix = self.format_matrix(swof_matrix)
    #         self.insert_matrix(output_file, formatted_swof_matrix, self.swof_from, self.swof_to)

    #         # sgof_matrix = self.create_sgof_matrix()
    #         # formatted_sgof_matrix = self.format_matrix(sgof_matrix)
    #         # self.insert_matrix(output_file, formatted_sgof_matrix, self.sgof_from, self.sgof_to)

    #         # pvdg_matrix, pvdg_df = self.create_pvdg_matrix()
    #         # formatted_pvdg_matrix = self.format_matrix(pvdg_matrix)
    #         # self.insert_matrix(output_file, formatted_pvdg_matrix, self.pvdg_from, self.pvdg_to)

    #         # pvto_matrix, pvto_df = self.create_pvto_matrix()
    #         # formatted_pvto_matrix = self.format_pvto_matrix(pvto_matrix)
    #         # self.insert_matrix(output_file, formatted_pvto_matrix, self.pvto_from, self.pvto_to)

    #         # equil_matrix, equil_df = self.create_equil_matrix()
    #         # formatted_equil_matrix = self.format_matrix(equil_matrix)
    #         # self.insert_matrix(output_file, formatted_equil_matrix, self.equil_line, self.equil_line)

    #         iter_df = pd.DataFrame({'iteration': [i] * 15})
    #         final_iter_df = pd.concat([swof_df, 
    #                                 #    pvdg_df, pvto_df, equil_df, 
    #                                    iter_df], axis=1)
    #         final_df = pd.concat([final_df, final_iter_df])

    #     return final_df

    def make_files(
        self
    ):
        final_df = pd.DataFrame()
        counter = 0
        for nw in range(1,6):
            for no in range(1,6):
                output_file = os.path.join(self.main_dir, f'SPE1CASE1_ITER_{counter}.DATA')
                shutil.copy2(self.src_file, output_file)
                iter_df = pd.DataFrame({'iteration': [counter] * 15, 'N_w': [nw]*15, 'N_o': [no]*15})
                swof_matrix, swof_df = self.create_swof_matrix(N_w=nw, N_o=no)
                formatted_swof_matrix = self.format_matrix(swof_matrix)
                self.insert_matrix(output_file, formatted_swof_matrix, self.swof_from, self.swof_to)
                final_iter_df = pd.concat([swof_df, iter_df], axis=1)
                final_df = pd.concat([final_df, final_iter_df])

                counter += 1

        return final_df


if __name__ == '__main__':
    import random
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
    # _, df = generator.create_swof_matrix(N_w=random.randint(1, 5), N_o=random.randint(1, 5))
    # print(df)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    print(generator.make_files())

    