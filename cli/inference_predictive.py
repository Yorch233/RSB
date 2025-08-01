import argparse
import os

import torch
import torchaudio
from tqdm import tqdm
from safetensors.torch import load_model

from RSB.dataset.AudioFolder import AudioFolder
from RSB.backbone import BackboneRegister
from RSB.dataset.ComplexSpecDatatet import STFTUtil

def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing configuration parameters for inference.

    Arguments:
        sampling_method (str): Sampling method (default: "SDE solver").
        skip_type (str): Step skipping strategy during sampling (default: "time uniform").
        NFE (int): Number of function evaluations (default: 3).
        ot_ode (bool): Whether to use ODE solver (default: False).
        target_path (str): Path to input audio file or directory containing noisy audios.
        local_rank (int): CUDA device index for distributed inference (default: 0).
        target_sample_rate (int): Output audio sample rate (default: 16000).
        model_dir (str): Directory containing pre-trained model checkpoints (default: "checkpoint").
        output_dir (str): Directory to save enhanced audios (default: "output").
        no_log (bool): Disable progress logging (default: False).
    """
    parser = argparse.ArgumentParser(description="Run inference of predictive model.")

    # Input/Output parameters
    parser.add_argument("--dataset", type=str, default='voicebank')
    parser.add_argument("--predictive_backbone", type=str, default='ncsnpp_base')
    parser.add_argument("--audio_path", type=str, help="Directory to noisy audios/ Path to noisy audio")
    parser.add_argument("--local_rank", type=int, default=0, help="CUDA Device index for inference")
    parser.add_argument(
        "--target_sample_rate",
        type=int,
        default=16000,
        help="The customized sample rate of enhanced audio",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the enhanced audios.",
    )
    parser.add_argument("--no_log", action="store_true", help="Whether to log.")

    return parser.parse_args()


def run_inference(args):
    """Execute audio enhancement using Schrodinger Bridge model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing configuration parameters.
    """
    # Validate input path existence
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Path does not exist, please check the parameter `audio_path`.")
    device = torch.device(args.local_rank)
    
    preditive_model = BackboneRegister.fetch(args.predictive_backbone)(input_channels=2, discriminative=True)
    predictive_checkpoint_path = os.path.join(
        'pretrained_predictive_model', args.dataset, f'{args.predictive_backbone}.pt')
    if not os.path.exists(predictive_checkpoint_path):
        raise RuntimeError(
            f"The discriminator checkpoint at path '{predictive_checkpoint_path}' does not exist. Please ensure the path is correct or the discriminator has been pre-trained."
        )
    checkpoint = torch.load(predictive_checkpoint_path, map_location='cpu')
    preditive_model.load_state_dict(checkpoint)
    preditive_model.to(device)
    preditive_model.eval()

    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process directory of audios
    if os.path.isdir(args.audio_path):
        # Load dataset and create data loader
        noisy_audios = AudioFolder(args.audio_path, sample_rate=16000, return_path=True)
        dataloader = noisy_audios.get_data_loader()

        with torch.no_grad():
            # Iterate over dataset with optional progress bar
            iter = (dataloader if args.no_log else tqdm(dataloader, desc=f"Enhancing Audios"))
            for _audio, _audio_name in iter:
                audio, audio_name = _audio.squeeze(0), _audio_name[0]
                audio_path = os.path.join(args.output_dir, f"{audio_name}.wav")

                y, invert_fn = STFTUtil.to_stft(audio, device=device)
                # Perform enhancement and save result
                x = preditive_model(y)
                enhanced_audio = invert_fn(x)
                torchaudio.save(
                    audio_path,
                    enhanced_audio.type(torch.float32).cpu().squeeze().unsqueeze(0),
                    args.target_sample_rate,
                )

        # Log completion if enabled
        if not args.no_log:
            print(
                f"Inference process has successfully completed. Enhanced audios are saved to \033[95m{args.output_dir}\033[0m."
            )

    # Process single audio file
    else:
        # Load and resample audio if necessary
        audio, sr = torchaudio.load(args.audio_path)
        if sr != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)

        # Generate output path
        audio_path = os.path.join(
            args.output_dir,
            f"{os.path.splitext(os.path.basename(args.audio_path))[0]}_enhanced.wav",
        )

        # Perform enhancement and save result
        y, invert_fn = STFTUtil.to_stft(audio, device=device)
         # Perform enhancement and save result
        x = preditive_model(y)
        enhanced_audio = invert_fn(x)
        torchaudio.save(
            audio_path,
            enhanced_audio.type(torch.float32).cpu().squeeze().unsqueeze(0),
            args.target_sample_rate,
        )

        # Log completion if enabled
        if not args.no_log:
            print(f"Enhanced audio is saved to \033[95m{args.output_dir}\033[0m.")


if __name__ == "__main__":
    # Parse command-line arguments and execute inference
    args = parse_args()
    run_inference(args)