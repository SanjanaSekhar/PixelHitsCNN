from ClusterConverter import ClusterConverter
import subprocess
from os import listdir, remove
from os.path import isfile, join, exists
import h5py
import shutil

class InputMaker:

    def __init__(self,input_folder,output_folder,dataset,template_id):
        #Dataset should be one of the layer keys in ClusterConverterConfig.json file
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.train_file =  join(self.input_folder, f"{dataset}_training.out")
        self.test_file = join(self.input_folder, f"{dataset}_testing.out")
        self.dataset = dataset
        self.template_id = template_id


    def unzip_dir(self):
        '''
        Unzips files from self.input_folder if skip_unzip is not set to false
        Creates a list of train and test files and extracts template info (pixelsize)
        '''
        dir_to_process = self.input_folder
        print(f"Processing {dir_to_process}")
        #Assumes that the .gz files will be template*gz
        onlyfiles = [f for f in listdir(dir_to_process) if (isfile(join(dir_to_process, f)) and "template" in f)]

        print("Unzipping")
        for i,file_name in enumerate(onlyfiles):
            if(i%100==0):
                print("{0}/{1}".format(i,len(onlyfiles)))
            file_name_out = file_name.replace(".gz","")
            subprocess.call(f"gunzip -c {dir_to_process}/{file_name} > {file_name_out}",shell=True)

        for i in range(len(onlyfiles)):
            onlyfiles[i] = onlyfiles[i].replace(".gz","")

        test_frac = 0.15
        n_test    = int(test_frac*len(onlyfiles))

        files_for_test  = onlyfiles[0:n_test]
        files_for_train = onlyfiles[n_test:]

        file_with_tpl_info = subprocess.check_output("grep Dot template*out",shell=True,encoding='UTF-8')
        file_with_tpl_info = file_with_tpl_info.split(":")[0]
        if file_with_tpl_info in files_for_test: 
            print("Removing extra file from testing")
            files_for_test.remove(file_with_tpl_info)
        if file_with_tpl_info.split("/")[-1] in files_for_train:
            print("Removing extra file from training")
            files_for_train.remove(file_with_tpl_info)

        with open(file_with_tpl_info,"r") as f:
            tpl_info = [next(f) for _ in range(2)]

        #Write the template info to the training and testing files
        with open(self.train_file,"w") as f:
            for line in tpl_info:
                f.write(line)
        with open(self.test_file,"w") as f:
            for line in tpl_info:
                f.write(line)

        print("Merging training files")
        for i,file_name in enumerate(files_for_train):
            if(i%100==0):
                print("{0}/{1}".format(i,len(files_for_train)))
            subprocess.call(f"cat {file_name} >> {self.train_file}",shell=True)

        print("Merging testing files")
        for i,file_name in enumerate(files_for_test):
            if(i%100==0):
                print("{0}/{1}".format(i,len(files_for_test)))
            subprocess.call(f"cat {file_name} >> {self.test_file}",shell=True)

        for file_name in onlyfiles:
            subprocess.call(f"rm {file_name}",shell=True)

    def convert_txt_files(self):
        clu_converter = ClusterConverter("ClusterConverterConfig.json", self.dataset,self.template_id)
        self.output_train = join(self.output_folder, f"{self.dataset}_train.hdf5")
        self.output_test = join(self.output_folder, f"{self.dataset}_test.hdf5")

        if exists(self.train_file):
            clu_converter.text_to_hdf5(self.train_file, self.output_train)
            print(f"Converted {self.train_file} to {self.output_train}")
        else:
            print(f"Warning: Train file not found: {self.train_file}")

        if exists(self.test_file):
            clu_converter.text_to_hdf5(self.test_file, self.output_test)
            print(f"Converted {self.test_file} to {self.output_test}")
        else:
            print(f"Warning: Test file not found: {self.test_file}")

        print("Conversion process completed.")
    
    def clear(self):
        subprocess.call(f"rm {self.train_file} {self.test_file}",shell=True)


if __name__ == "__main__":
    #For testing
    test_object = InputMaker("/uscms_data/d3/roguljic/NN_CPE/clusters/template_events_d83201_d83400/","/uscms_data/d3/roguljic/NN_CPE/clusters/","L1U")
    test_object.unzip_dir()
    test_object.convert_txt_files()    
    test_object.clear()