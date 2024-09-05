import numpy as np
import re

class neural_network:
    
    def __init__(self, input_nodes, output_nodes, vol_sources, i_sources, v_diode_pos, v_diode_neg, amp, camp):        
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.vol_sources = vol_sources
        self.size_of_layers = size_of_layers
        self.isources = i_sources
        self.vdiode_pos = v_diode_pos
        self.vdiode_neg = v_diode_neg
        self.amp = amp
        self.camp = camp
        self.parameters = []
    
    def build_inputs(self, input_nodes, vol_sources):
        lines = []
        for i, (node, source) in enumerate(zip(input_nodes, vol_sources)):
            line = f"VSOURCE{i+1} {node} 0 {source} \n"
            lines.append(line)
            self.parameters.append(source)
        return lines
    
    def build_nudge_sources(self, output_nodes, i_sources):
        lines = []
        for i, (node, source) in enumerate(zip(output_nodes, i_sources)):
            line = f"ISOURCE{i+1} {node} 0 {source} \n"
            lines.append(line)
            self.parameters.append(source)
        return lines        
    def synaptic_layer(self, input_nodes, output_nodes, layer):
        lines = []
        for in_node in input_nodes:
            for out_node in output_nodes:
                in_node_int = self.extract_number(in_node)
                out_node_int = self.extract_number(out_node)
    
                parameter = f"R_{layer}_{in_node_int}{out_node_int}"
                self.parameters.append(parameter)
                line = f"R{layer}{in_node_int}{out_node_int} {in_node} {out_node} {parameter}\n"
                lines.append(line)
        return lines
    
    def neuron_layer(self, inputs_2_neurons, outputs_2_neurons, layer):
        lines = []
        for i, (in_node, out_node) in enumerate(zip(inputs_2_neurons, outputs_2_neurons)):
            in_node_int = self.extract_number(in_node)
            out_node_int = self.extract_number(out_node)
            line = f"XI{layer}{in_node_int}{out_node_int} {in_node} {out_node} NEURON \n"
            lines.append(line)
        return lines
    
    def extract_number(self, node):
        matches = re.findall(r'\d+', node)  # Find all integer matches
        return matches[-1] if matches else ''  # Return the last match if available
    

    def build_layers_automatically(self):
        # Get all node names from the structured function
        node_names = self.built_node_names()

        # Lists to store details of all layers for verification or further operations
        input_layers = self.build_inputs(self.input_nodes, self.vol_sources)
        output_sources = self.build_nudge_sources(self.output_nodes, self.isources)
        all_synaptic_layers = []
        all_neuron_layers = []

        # Iterate over node_names with a step of 2 to handle pairs correctly
        for i in range(0, len(node_names) - 1, 2):
            # Synaptic layer between this layer's outputs and the next layer's inputs
            input_nodes = node_names[i]
            output_nodes = node_names[i + 1]
            synaptic_layer = self.synaptic_layer(input_nodes, output_nodes, layer=i//2)
            all_synaptic_layers.append(synaptic_layer)

            # Check if there is a subsequent pair to form a neuron layer
            if i + 2 < len(node_names):
                # Neuron layer between this pair's output and the next pair's input
                neuron_layer = self.neuron_layer(output_nodes, node_names[i + 2], layer=i//2 + 1)
                all_neuron_layers.append(neuron_layer)

        return input_layers, all_synaptic_layers, all_neuron_layers, output_sources
    
    def build_parameters(self, mode):
        lines = []
        
        if mode in ["res", "all"]:
            for parameter in self.parameters:
                line = f".PARAM {parameter}=100\n"
                lines.append(line)
        
        if mode in ["amp", "all"]:
            amp = self.amp
            camp = self.camp
            line1 = f".PARAM AMP={amp}\n"
            lines.append(line1)
            line2 = f".PARAM AMPC={camp}\n"
            lines.append(line2)
        
        if mode in ["non_lin", "all"]:
            vdiode2 = self.vdiode_neg
            vdiode1 = self.vdiode_pos
            line1 = f".PARAM VDIODE1={vdiode1}\n"
            lines.append(line1)
            line2 = f".PARAM VDIODE2={vdiode2}\n"
            lines.append(line2)
      
        if mode in ["isources", "all"]:
            for isource in self.isources:
                line = f".PARAM {isource}=0\n"
                lines.append(line)
         
                
        if mode in ["vsources", "all"]:
            for vsource in self.vol_sources:
                line = f".PARAM {vsource}=0\n"
                lines.append(line)
            
         
        lines.append(".PARAM FORM=0\n")
        return lines
    
    def build_node_names(self, n_of_inputs, template):
        node_names = []      
        for i in range(1, n_of_inputs+1):
            node_name = f"{template}{i}"
            node_names.append(node_name)
        return node_names
    
    def build_network(self):
        
        
        input_nodes = self.input_nodes
        vol_sources = self.vol_sources
        sourcess = self.build_inputs(input_nodes, vol_sources)

        
        layer = 0
        template = "V_0_IN"
        input_nodes = self.build_node_names(4, template)
        
        
        #Sources    
        
        template = "V_0_N_IN"
        output_nodes = self.build_node_names(4, template)
        first_synapses = self.synaptic_layer(self.input_nodes, output_nodes, layer)
        
        layer = 1
        template = "V_0_N_IN"
        inputs_2_neurons = self.build_node_names(4, template)
        template = "V_0_N_OUT"
        outputs_2_neurons = self.build_node_names(4, template)
        first_hidden = self.neuron_layer(inputs_2_neurons, outputs_2_neurons, layer)
        
        layer = 2
        template = "V_0_N_OUT"
        input_nodes = self.build_node_names(8, template)
        template = "V_1_N_IN"
        output_nodes = self.build_node_names(4, template)
        second_synapses = self.synaptic_layer(input_nodes, output_nodes, layer)
        
        template = "V_1_N_IN"
        inputs_2_neurons = self.build_node_names(4, template)
        template = "V_2_N_OUT"
        outputs_2_neurons = self.build_node_names(4, template)
        second_hidden = self.neuron_layer(inputs_2_neurons, outputs_2_neurons, layer)
        
        layer = 3
        template = "V_2_N_OUT"
        input_nodes = self.build_node_names(4, template)
        template = "V_Y"
        output_nodes = self.build_node_names(2, template)
        third_synapses = self.synaptic_layer(input_nodes, output_nodes, layer)
        
        
        nudge_sources = self.build_nudge_sources(output_nodes, self.isources)
        
        # Combine all lines
        network_description =  first_synapses + first_hidden + second_synapses + second_hidden + third_synapses + sourcess + nudge_sources
        
        return network_description
    def built_node_names(self):
        node_names = []
        vol_sources = self.vol_sources
        input_nodes = self.input_nodes
        output_nodes = self.output_nodes
        sourcess = self.build_inputs(input_nodes, vol_sources)
        
        num_layers = len(self.size_of_layers)  # this include just the size of the hidden layers
        
        # Handling for the first layer separately if needed
        if num_layers > 0:
            template = "VIN"
            input_nodes = self.build_node_names(len(input_nodes), template)
            node_names.append(input_nodes)

        ## Builds synaptic connections
        for i, layer_size in enumerate(self.size_of_layers):
            # Define templates for input and output nodes for each layer

    
            # Check if it's the last iteration
            if i == len(self.size_of_layers) - 1:
                # Special handling for the last layer
                special_template = f"V_Y"
                special_nodes = self.build_node_names(layer_size, special_template)
                node_names.append(special_nodes)
            else:
                # Normal handling for other layers
                input_template = f"V_{i}_N_IN"
                input_nodes = self.build_node_names(layer_size, input_template)
                node_names.append(input_nodes)
                output_template = f"V_{i}_N_OUT"
                output_nodes = self.build_node_names(self.size_of_layers[i], output_template)
                node_names.append(output_nodes)
    
        return node_names
        
        
    def built_network_advanced(self):
        vol_sources = self.vol_sources
        input_nodes = self.input_nodes
        output_nodes = self.output_nodes
        sourcess = self.build_inputs(input_nodes, vol_sources)
        
        num_layers = len(self.size_of_layers)
        ##builts synaptic connections
        for i in enumerate(self.size_of_layers): #size of layers is a list 
            layer = i
            if layer == 0:
                input_template = "VIN"
            else:
                input_template = f"V_{i-1}_N_OUT"
                
            input_nodes = self.build_node_names(self.size_of_layers[i], input_template)
            output_template = f"V_{i}_N_IN"
            output_nodes = self.build_node_names(self.size_of_layers[i+1], output_template)
            
            if layer+1 == num_layers:
                output_template = "V_Y"
                
            output_nodes = self.build_node_names(self.size_of_layers[i+1], output_template)




    def write_to_file(self, file_name):
        # Generate the network description and parameters
        network_description = self.build_network()
        parameter_lines = self.build_parameters("all")

        # Define the content structure
        header = "***\n" \
                 "*** Generated for: eldoD\n" \
                 "*** Generated on: xxxx xxxx xxxx\n" \
                 "*** Design library name: tests\n" \
                 "*** Design cell name: kendal_non_linear_moons_easy\n" \
                 "*** Design view name: schematic\n" \
                 ".GLOBAL\n"

        mid_sect = ".LIB /cao/DK/ST/HCMOS9A_10.9/Addon_NVM_H9A@2018.4.1/tools/eldo/model_oxram/OxRRAM.lib OxRRAM_TT\n" \
                   ".LIB /home/filip/CMOS130/corners.eldo\n" \
                   ".LIB /home/filip/Documents/MyDiode.lib\n\n" \
                   "*** Library name: tests\n" \
                   "*** Cell name: neuron\n" \
                   "*** View name: schematic\n" \
                   ".SUBCKT NEURON VIN VOUT\n" \
                   "    D0 VIN NET6 diode1\n" \
                   "    D1 NET7 VIN diode1\n" \
                   "    V2 NET6 0 DC VDIODE1\n" \
                   "    V3 NET7 0 DC VDIODE2\n" \
                   "    F0 0 VIN EVCVS1 {AMPC}\n" \
                   "    EVCVS1 VOUT 0 VIN 0 AMP\n" \
                   ".ENDS\n" \
                   "*** End of subcircuit definition.\n\n" \
                   "*** Library name: tests\n" \
                   "*** Cell name: kendal_non_linear_moons_easy\n" \
                   "*** View name: schematic\n"

        simulation_details = ".OP\n" \
                             ".DC\n" \
                             ".PROBE V\n" \
                             ".END\n"

        # Open the file and write the contents
        with open(file_name, 'w') as file:
            file.write(header)
            for line in parameter_lines:
                file.write(line)
            file.write(mid_sect)
            for line in network_description:
                file.write(line)
            file.write(simulation_details)
        print(f"Network description and parameters have been saved to {file_name}.")
        
        
if __name__ == "__main__":
    # Define input parameters
    input_nodes = ["VIN1", "VIN2", "VIN3", "VIN4", "VIN5", "VIN6", "VIN7", "VIN8"]
    output_nodes = ["VY_1", "VY_2"]  # Assuming you need output nodes; adjust as necessary
    vol_sources = ["VDC1", "VDC2", "VDC3", "VDC4", "VDC5", "VDC6", "VDC7", "VDC8"]
    i_sources = ["INUDGE1", "INUDGE2"]  # Updated to match your `i_sources` format
    size_of_layers = [2,2] #sources don't count as a layer
    v_diode_pos = 0.5
    v_diode_neg = -0.5
    amp = 1
    camp = 1

    # Create an instance of the neural_network class
    nn = neural_network(input_nodes, output_nodes, vol_sources, i_sources, v_diode_pos, v_diode_neg, amp, camp)

    # Write the network description and parameters to a .cir file
    nn.write_to_file("/home/filip/simulations/sample_files/eldo_samples/python_generated_netlists/network.cir")
    nodess = nn.built_node_names()
    all_layers = nn.build_layers_automatically()
    full_parameters = nn.build_parameters("all")
    nn.write_to_file("/home/filip/simulations/sample_files/eldo_samples/python_generated_netlists/network.cir")
