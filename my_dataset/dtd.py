import os
from .utils import Datum, DatasetBase, listdir_nohidden

template = ['{} texture.']


class DescribableTextures(DatasetBase):

    dataset_dir = 'dtd'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')

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
        # The data are supposed to be organized into the following structure
        # =============
        # images/
        #     category1/
        #     category2/
        #     ...
        # =============
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        def _collate(ims):
            items = []
            for im in ims:
                item = Datum(
                    impath=im,
                    label=0,  # OOD数据集不需要真实标签，统一设为0
                    classname='ood'  # 统一类名
                )
                items.append(item)
            return items

        train, val, test = [], [], []
        for category in categories:
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            
            # 简单地将所有数据作为测试集（OOD场景通常不需要训练/验证集）
            test.extend(_collate(images))
        
        # 对于OOD数据集，我们只需要测试集
        return [], [], test
