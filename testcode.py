import os
def get_sample_files(classpath):
    # classpath: ex: iamges/train, which includes different classes samples with each class in its folder
    rootpath=os.path.abspath('.')
    classpath=os.path.join(rootpath,classpath)
    class_list=os.listdir(classpath)
    class_list.sort()
    number_of_class=len(class_list)
    samples=[]
    for i in class_list:
        filesdir=os.path.join(classpath,i)
        files=os.listdir(filesdir)
        for j in range(len(files)):
            files[j]=os.path.join(filesdir,files[j])
        samples+=files
    return class_list,number_of_class,samples

if __name__=='__main__':
    get_sample_files("ALL_hd_random\\train")