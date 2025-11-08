import os
import gdown

def download_from_drive(file_id, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, filename)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"downloading: {filename}")
    gdown.download(url, output_path, quiet=False)
    print(f"Saved: {output_path}\n")
    return output_path

if __name__ == "__main__":
    save_dir = r"./pretrained_model"
    files = [
        ("1J02zklnEsizdxiasg6zbVmEftz9BYvzQ", "audio_mdl.pth"),
        ("1gSRRh-HpDE6zCN-IF0ZyKvfG4cb_ieBG", "finetuned_panns.pth"),
        ("1MKtbIuO1r8B5RjIOn2wSSRSoQ0VvEOzX","finetuned_vggish.pt"),
    ]
    for fid, fname in files:
        download_from_drive(fid, save_dir, fname)

    print("Completed all downloads.")


