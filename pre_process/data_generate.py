import os
from datetime import datetime
import numpy as np

from abaqus import *
from abaqusConstants import *
from odbAccess import openOdb

# Configs
material = "Polynomial"
data_type_set = ["train", "test1", "test2", "test_single_shear", "test_single_tensile"]
odb_path_prefix = r"D:\Courses\Dissertation\ABAQUS models\Materials"
export_path_prefix = r"C:\Users\Xinyi\PycharmProjects\NN-Hyperelasticity\data"
dim = 3
node_per_element = 4


def data_generate(odb_path, export_path):
    # Message to the terminal
    time = datetime.now()
    print("=" * 100)
    print(time)
    print("Start generating data...")
    print("Data source path:            " + odb_path)
    print("Export path:                 " + export_path)
    print("Dimension:                   " + str(dim))
    print("Node per element:            " + str(node_per_element))
    summary_file.write("=" * 100 + '\n')
    summary_file.write(str(time) + '\n')
    summary_file.write("Start generating data...\n")
    summary_file.write("Data source path:            " + odb_path + '\n')
    summary_file.write("Export path:                 " + export_path + '\n')
    summary_file.write("Dimension:                   " + str(dim) + '\n')
    summary_file.write("Node per element:            " + str(node_per_element) + '\n')

    # Open odb file
    odb = openOdb(odb_path, readOnly=True)

    # Assembly of FEM model
    # Elements
    element_set = odb.rootAssembly.elementSets[' ALL ELEMENTS'].instances[0].elements
    num_element = len(element_set)
    print("-" * 100)
    print("Number of elements:          " + str(num_element))
    summary_file.write("-" * 100 + '\n')
    summary_file.write("Number of elements:          " + str(num_element) + '\n')
    element_array = np.zeros((num_element, node_per_element), dtype=int)
    for rank_frame, element in enumerate(element_set):
        element_array[rank_frame] = element.connectivity
    element_array -= 1
    np.savetxt(export_path + r"\element.csv", element_array, delimiter=',', header='a,b,c,d', comments='')

    # Nodes
    node_set = odb.rootAssembly.elementSets[' ALL ELEMENTS'].instances[0].nodes
    num_node = len(node_set)
    print("-" * 100)
    print("Number of nodes:             " + str(num_node))
    summary_file.write("-" * 100 + '\n')
    summary_file.write("Number of nodes:             " + str(num_node) + '\n')
    node_array = np.zeros((num_node, dim), dtype=float)
    for node in node_set:
        node_array[node.label - 1] = node.coordinates
    np.savetxt(export_path + r"\node.csv", node_array, delimiter=',', header='x,y,z', comments='')

    # Boundary condition
    boundary_set = odb.rootAssembly.surfaces
    boundary_keys = boundary_set.keys()
    num_boundary = len(boundary_keys)
    print('-' * 100)
    print("Number of boundaries:        " + str(num_boundary))
    print("Names of boundaries: ")
    print(boundary_keys)
    summary_file.write('-' * 100 + '\n')
    summary_file.write("Number of boundaries:        " + str(num_boundary) + '\n')
    summary_file.write("Names of boundaries: \n")
    summary_file.write(str(boundary_keys) + '\n')
    boundary_array = np.zeros((num_node, num_boundary), dtype=int)
    for rank_frame, boundary_key in enumerate(boundary_keys):
        boundary = boundary_set[boundary_key]
        boundary_node_set = boundary.nodes[0]
        for node in boundary_node_set:
            boundary_array[node.label - 1, rank_frame] = 1
    np.savetxt(export_path + r"\bc.csv", boundary_array, delimiter=',',
               header=','.join(["Set-" + str(i + 1) for i in range(num_boundary)]), comments='')

    # Boundary for test
    boundary_test_set = odb.rootAssembly.nodeSets
    boundary_test_keys = boundary_test_set.keys()
    boundary_test_keys.remove(' ALL NODES')
    num_boundary_test = len(boundary_test_keys)
    print('-' * 100)
    print("Number of test boundaries:   " + str(num_boundary_test))
    print("Names of test boundaries: ")
    print(boundary_test_keys)
    summary_file.write('-' * 100 + '\n')
    summary_file.write("Number of test boundaries:   " + str(num_boundary_test) + '\n')
    summary_file.write("Names of test boundaries: \n")
    summary_file.write(str(boundary_test_keys) + '\n')
    boundary_test_array = np.zeros((num_node, num_boundary_test), dtype=int)
    for rank_frame, boundary_test_key in enumerate(boundary_test_keys):
        boundary_test = boundary_test_set[boundary_test_key]
        boundary_test_node_set = boundary_test.nodes[0]
        for node in boundary_test_node_set:
            boundary_test_array[node.label - 1, rank_frame] = 1
    np.savetxt(export_path + r"\bc_test.csv", boundary_test_array, delimiter=',',
               header=','.join(["Set-" + str(i + 1) for i in range(num_boundary_test)]), comments='')

    # Frames of FE simulation
    num_step = len(odb.steps)
    print("-" * 100)
    print("Number of steps:             " + str(num_step))
    summary_file.write("-" * 100 + '\n')
    summary_file.write("Number of steps:             " + str(num_step) + '\n')
    for rank_step in range(num_step):
        frames = odb.steps['Step-' + str(rank_step + 1)].frames
        num_frame = len(frames)
        print("[Step-" + str(rank_step + 1) + "] Number of frames:   " + str(num_frame))
        summary_file.write("[Step-" + str(rank_step + 1) + "] Number of frames:   " + str(num_frame) + '\n')
        for rank_frame, frame in enumerate(frames):
            step_frame_ID = str(rank_step + 1) + '-' + str(frame.frameId)
            simu_outcome = frame.fieldOutputs
            if not os.path.exists(export_path + "\\" + step_frame_ID):
                os.makedirs(export_path + "\\" + step_frame_ID)

            # Displacement
            disp_outcome = simu_outcome['U'].values
            disp_array = np.zeros((num_node, 3), dtype=float)
            for i in range(num_node):
                disp_array[i] = disp_outcome[i].dataDouble
            np.savetxt(export_path + "\\" + step_frame_ID + "\\" + "disp.csv", disp_array,
                       delimiter=',', header='ux,uy,uz', comments='')

            # Stress
            stress_outcome = simu_outcome['S'].values
            stress_array = np.zeros((num_element, 7), dtype=float)
            for i in range(num_element):
                stress_array[i, 0:6] = stress_outcome[i].dataDouble
                stress_array[i, 6] = stress_outcome[i].mises
            np.savetxt(export_path + "\\" + step_frame_ID + "\\" + "stress.csv", stress_array,
                       delimiter=',', header='sxx,syy,szz,sxy,sxz,syz,smises', comments='')

            # Reaction force
            rf_outcome = simu_outcome['RF'].values
            rf_array = np.zeros((num_node, 3), dtype=float)
            global_rf_array = np.zeros((num_boundary, 3), dtype=float)
            for i in range(num_node):
                rf_array[i] = rf_outcome[i].dataDouble
            np.savetxt(export_path + "\\" + step_frame_ID + "\\" + "node_rf.csv", rf_array,
                       delimiter=',', header='fx,fy,fz', comments='')
            for i in range(num_boundary):
                boundary_indices = (boundary_array[:, i] == 1)
                global_rf_array[i, :] = np.sum(rf_array[boundary_indices], axis=0)
            np.savetxt(export_path + "\\" + step_frame_ID + "\\" + "global_rf.csv", global_rf_array,
                       delimiter=',', header='fx,fy,fz', comments='')

    # Close odb file
    odb.close()

    # Message to the terminal
    print("All done!")
    print("=" * 100)
    summary_file.write("All done!\n")
    summary_file.write("=" * 100 + '\n')


if __name__ == '__main__':
    for data_type in data_type_set:
        odb_path = odb_path_prefix + "\\" + material + "\\" + data_type + "\\" + "Job-1.odb"
        export_path = export_path_prefix + "\\" + material + "\\" + data_type
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        with open(export_path + r"\log.txt", 'w') as summary_file:
            data_generate(odb_path, export_path)
