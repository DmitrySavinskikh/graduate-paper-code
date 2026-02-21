import random
import os
import shutil

class MatrixGenerator():
    def __init__(
            self,
            iters = 1,
            src_file= '',
            main_dir= ''
        ):
        self.iters = iters
        self.src_file = src_file
        self.main_dir = main_dir

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

    def format_matrix(
            self,
            matrix
        ):
        output = []
        for row in matrix:
            formatted_row = '\t'.join([f"{value:.6f}" for value in row]) # ai / it's for match with format in orig file
            output.append(formatted_row)
        output[-1] = output[-1] + ' /'
        
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

            # # Заменяем строки
            # lines[from_:to_ + 1] = [line + '\n' for line in input_block]
            
            # # Записываем обратно
            # with open(file_path, 'w') as f:
            #     f.writelines(lines)

    def make_files(
            self,
            matrix_kind: str
        ):

        for i in range(self.iters):
            output_file = os.path.join(self.main_dir, f'SPE1CASE1_ITER_{i}.DATA')
            shutil.copy2(self.src_file, output_file)

            if matrix_kind == 'random':
                matrix = self.random_matrix()
            elif matrix_kind == 'min2max': # only for self.iters = 1
                matrix = self.min2max_matrix()
            formatted_matrix = self.format_matrix(matrix)
            self.insert_matrix(output_file, formatted_matrix, 144, 158)




if __name__ == '__main__':
    generator = MatrixGenerator(
        iters=2, 
        src_file='/home/dmitrysavinskikh/data/opm-data/spe1/SPE1CASE1.DATA', 
        main_dir='/home/dmitrysavinskikh/graduate-paper-code/spe1')
    generator.make_files(matrix_kind='random')

    