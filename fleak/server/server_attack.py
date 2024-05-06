from .server import Server
from ..attack.DLG import dlg, idlg
from ..attack.IG import ig_single, ig_multi
from ..attack.robbing_the_fed import robbing


class ServerAttacker(Server):

    def __init__(
        self,
        server_id=None,
        server_group=None,
        global_model=None,
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
            secret = dict(weight_idx=0, bias_idx=1, shape=(3, 32, 32), structure="cumulative")
            secrets = {"ImprintBlock": secret}
            dummy_data, dummy_label = robbing(self.global_model, local_grads, secrets, self.device)
            self.dummy.append(dummy_data)
            self.dummy.append_label(dummy_label)
        else:
            raise ValueError("Unexpected {} Attack Type.".format(method))