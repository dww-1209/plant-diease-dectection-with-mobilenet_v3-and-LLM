def read_txt_to_dict(file_path):
    ### 植物类型
    Plant_class= {0:'苹果',1:'樱桃',2:'玉米',3:'葡萄',4:'柑桔',5:'桃',6:'辣椒',7:'马铃薯',8:'草莓',9:'番茄'}
    ### 是否健康
    Healthy = {0:'未患病',1:'患病'}
    ### 病变程度
    Diseased_degree = {0:'健康',1:'一般',2:'严重',3:'患病但不分程度'}
    ### 病害类型
    Diseased_class = {0:'健康',1:'苹果黑星病',2:'苹果灰斑病',3:'苹果雪松锈病',4:'樱桃白粉病',5:'玉米灰斑病',6:'玉米锈病',7:'玉米叶斑病',
                      8:'玉米花叶病毒病',9:'葡萄黑腐病',10:'葡萄轮斑病',11:'葡萄褐斑病',12:'柑桔黄龙病',13:'桃疮痂病',14:'辣椒疮痂病',
                      15:'马铃薯早疫病',16:'马铃薯晚疫病',17:'草莓叶枯病',18:'番茄白粉病',19:'番茄疮痂病',20:'番茄早疫病',
                      21:'番茄晚疫病',22:'番茄叶霉病',23:'番茄斑点病',24:'番茄斑枯病',25:'番茄红蜘蛛损伤',26:'番茄黄化曲叶霉病',27:'番茄花叶病毒病'}
    data_dict = {}
    
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read each line in the file
        for line in file:
            # Split the line into numbers
            numbers = line.strip().split()
            
            # The first number is the key
            key = int(numbers[0])
            
            # The remaining numbers are the values (as a list of integers)
            idx = list(map(int, numbers[1:]))
            values = []

            #植物类型
            values.append(Plant_class[idx[0]])
            #是否健康
            values.append(Healthy[idx[1]])
            #病变程度
            values.append(Diseased_degree[idx[2]])
            #病害类型
            values.append(Diseased_class[idx[3]])

            data_dict[key] = values
    
    return data_dict
