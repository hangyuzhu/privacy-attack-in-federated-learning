from typing import Callable
from fleak.attack import dmgan


def dmgan_method(method: Callable):
    def inner(self):
        # gan attack first
        dmgan(
            tracked_class=self.tracked_class,
            generator=self.generator,
            discriminator=self.client_model,
            dataloader=self.train_loader,
            dummy=self.dummy,
            rec_epochs=self.rec_epochs,
            d_optimizer=self.d_optimizer,
            g_optimizer=self.g_optimizer,
            criterion=self.criterion,
            noise_dim=100,
            device=self.device,
        )
        # perform normal federated training
        return method(self)
    return inner


def _class_decorator(cls, wrapper):
    for attr_name in dir(cls):
        if attr_name == "train":
            func = getattr(cls, attr_name)
            assert callable(func)
            wrapped = wrapper(func)
            setattr(cls, attr_name, wrapped)
    return cls


def dmgan_class(cls):
    return _class_decorator(cls, dmgan_method)