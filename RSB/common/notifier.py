# RSB/common/notifier.py
import logging

import requests

logger = logging.getLogger(__name__)


class WeChatNotifier:
    """Send WeChat notifications via AutoDL API."""

    API_URL = "https://www.autodl.com/api/v1/wechat/message/send"

    def __init__(self, token: str, run_name: str, enabled: bool = True):
        self.token = token
        self.run_name = run_name
        self.enabled = enabled

    def send(self, title: str, content: str) -> bool:
        if not self.enabled:
            return False
        try:
            resp = requests.post(
                self.API_URL,
                json={
                    "title": title,
                    "name": self.run_name,
                    "content": content,
                },
                headers={"Authorization": self.token},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info(f"WeChat notification sent: {title}")
                return True
            else:
                logger.warning(
                    f"WeChat notification failed ({resp.status_code}): {resp.text}"
                )
                return False
        except Exception as e:
            logger.warning(f"WeChat notification error: {e}")
            return False

    def send_training_start(
        self, dataset: str, bridge_type: str, training_method: str,
        num_epoch: int, batch_size: int, learning_rate: float,
    ):
        title = "Training Started"
        content = (
            f"Dataset: {dataset}\n"
            f"Bridge: {bridge_type} | Method: {training_method}\n"
            f"Epochs: {num_epoch} | BS: {batch_size} | LR: {learning_rate}"
        )
        return self.send(title, content)

    def send_epoch_metrics(
        self, epoch: int, pesq: float, sisdr: float, valid_loss: float,
        best_pesq: float, best_sisdr: float, best_valid_loss: float,
        early_stop_cnt: int, patience: int,
    ):
        title = f"Epoch {epoch} | Metrics"
        content = (
            f"PESQ={pesq:.4f}(best:{best_pesq:.4f})\n"
            f"SI-SDR={sisdr:.4f}(best:{best_sisdr:.4f})\n"
            f"Valid Loss={valid_loss:.6f}(best:{best_valid_loss:.6f})\n"
            f"Early-stop in {patience - early_stop_cnt} epoch(es)"
        )
        return self.send(title, content)

    def send_training_end(
        self, reason: str, final_epoch: int,
        best_pesq: float, best_sisdr: float,
    ):
        title = f"Training Finished | {reason}"
        content = (
            f"Stopped at epoch {final_epoch}\n"
            f"Best PESQ={best_pesq:.4f}\n"
            f"Best SI-SDR={best_sisdr:.4f}"
        )
        return self.send(title, content)
