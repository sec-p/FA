import os
from .utils import Datum, DatasetBase, listdir_nohidden

template = ['a photo of a {}.']


class INaturalist(DatasetBase):

    dataset_dir = 'iNaturalist'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir

        self.template = template

        # 直接从图像目录加载数据，不使用JSON分割文件
        train, val, test = self.read_and_split_data(self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)
    
    def read_and_split_data(
        self,
        image_dir,
        p_trn=0.5,
        p_val=0.2,
        ignored=[],
        new_cnames=None
    ):
        # 直接从图像目录加载数据，不使用JSON分割文件
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(
                    impath=im,
                    label=y, # is already 0-based
                    classname=c
                )
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            
            # 简单地将所有数据作为测试集（OOD场景通常不需要训练/验证集）
            test.extend(_collate(images, label, category))
        
        # 对于OOD数据集，我们只需要测试集
        return [], [], test