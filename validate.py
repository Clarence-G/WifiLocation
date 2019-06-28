def validate(method,truth,data,labellist,k,m):
    '''
    :param method: 判别方法，nn，knn，wknn
    :param truth: 测试数据编号
    :param data: 训练数据集
    :param labellist: 训练数据地址标签
    :param k:
    :param m:  扰动范围
    :return:
    '''
    disturbdata = data[truth]+m*np.random.random(len(data[truth]))  # 由于没有收集测试数据，在此加上扰动充当测试数据。
    predict = method(disturbdata,data,labellist,k)
    m,n = predict
    i,j = labellist[truth]
    print('use method {}, the predict result is {}'.format(method.__name__,predict))
    return np.linalg.norm((abs(m-i),abs(n-j)))




if __name__=='__main__':
    # 测试
    # 建立地点标签
    data = np.load('data/dataset.npy')
    labellist = []
    for i in range(len(data)):
        y, x  = divmod(i,6)  # 行x， 列 y
        labellist.append((x,y))
    # print(labellist)
    for truth in range(len(data)):
        print('the truth is {}'.format(labellist[truth]))
        validate(classifynn,truth,data,labellist,3,50)
        validate(classifyknn, truth, data, labellist,3,50)
        validate(classifywknn, truth, data,labellist, 3,50)