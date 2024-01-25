import base64
import io
import os
import time

import numpy as np
from ts.torch_handler.vision_handler import VisionHandler
from ts.handler_utils.timer import timed
from torchvision import transforms as T
import torch
from PIL import Image
import logging

from ts.torch_handler.base_handler import PROFILER_AVAILABLE


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


class U2NetHandler(VisionHandler):
    def load_images(self, data):
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))

            images.append(image)

        return images

    @timed
    def preprocess(self, images):
        trans = T.Compose([
            T.Resize(320),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        return torch.stack([trans(im) for im in images])

    @timed
    def postprocess(self, images, data):
        preds = data[0][:, 0, :, :]  # nchw
        predicts = normPRED(preds)
        predict_nps = predicts.cpu().detach().numpy()
        # masks = predict_nps.astype(np.uint8)
        masks = []

        for idx, pre in enumerate(predict_nps):
            alpha_channel = Image.fromarray(pre * 255).convert("L")
            alpha_channel = alpha_channel.resize(images[idx].size, resample=Image.BILINEAR)
            images[idx].putalpha(alpha_channel)
            buf = io.BytesIO()
            images[idx].save(buf, 'PNG')
            # base64_encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
            masks.append(buf.getvalue())
        return masks

    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artifacts parameters.

        Returns:
            list : Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        is_profiler_enabled = os.environ.get("ENABLE_TORCH_PROFILER", None)
        if is_profiler_enabled:
            if PROFILER_AVAILABLE:
                if self.manifest is None:
                    # profiler will use to get the model name
                    self.manifest = context.manifest
                output, _ = self._infer_with_profiler(data=data)
            else:
                raise RuntimeError(
                    "Profiler is enabled but current version of torch does not support."
                    "Install torch>=1.8.1 to use profiler."
                )
        else:
            if self._is_describe():
                output = [self.describe_handle()]
            else:
                images = self.load_images(data)
                data_preprocess = self.preprocess(images)
                if not self._is_explain():
                    output = self.inference(data_preprocess)
                    output = self.postprocess(images, output)
                else:
                    output = self.explain_handle(data_preprocess, data)

        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        return output
