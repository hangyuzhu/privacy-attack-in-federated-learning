from .server import Server
from ..attack import dlg, idlg
from ..attack import ig_single, ig_multi
from ..attack import invert_linear_layer
from ..attack import grnn
from ..attack import ggl


class ServerAttacker(Server):

    def __init__(
        self,
        server_id=None,
        server_group=None,
        global_model=None,
        generator=None,
        test_loader=None,
        dummy=None,
        local_epochs=1,
        local_lr=0.1,
        device=None
    ):
        super().__init__(
            server_id=server_id,
            server_group=server_group,
            global_model=global_model,
            test_loader=test_loader,
            device=device
        )
        if generator is not None:
            self.generator = generator.to(self.device)

        self.dummy = dummy
        self.local_epochs = local_epochs
        self.local_lr = local_lr

    def attack(self, method):
        """
        Randomly select a client to infer its private data

        :param method: attack method
        :return: reconstructed data and labels
        """
        local_grads = self.extract_gradients(self.updates[0][-1])
        # replace the global model by client model
        self.global_model.load_state_dict(self.updates[0][-1])

        if method == "dlg":
            dlg(self.global_model, local_grads, self.dummy, 300, self.device)
        elif method == "idlg":
            idlg(self.global_model, local_grads, self.dummy, 300, 1.0, self.device)
        elif method == "ig_single":
            ig_single(self.global_model, local_grads, self.dummy, 4000, 0.1, 1e-6, self.device)
        elif method == "ig_multi":
            ig_multi(
                self.global_model, local_grads, self.dummy,
                8000, 0.1, self.local_epochs, self.local_lr, 1e-6, self.device)
        elif method == "robbing":
            invert_linear_layer(local_grads, self.dummy)
        elif method == "ggl":
            ggl(self.global_model, self.generator, local_grads, self.dummy, 25000, self.device)
        elif method == "grnn":
            grnn(self.global_model, local_grads, self.dummy, 1000, 1e-3, self.device)
        else:
            raise ValueError("Unexpected {} Attack Type.".format(method))