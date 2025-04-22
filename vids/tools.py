import os
import sys
import ffmpeg

def convert_all_mov_files(input_dir, output_dir):
    """
    Convert all .mov files in input_dir to .mp4 files in output_dir.
    Re-encodes video to 30fps with H.264, CRF 23, and 'veryfast' preset.
    Removes all audio tracks.
    
    Args:
        input_dir (str): Directory containing .mov files
        output_dir (str): Directory where .mp4 files will be saved
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Get all .mov files in the input directory
    mov_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mov')]
    
    if not mov_files:
        print(f"No .mov files found in {input_dir}")
        return
    
    print(f"Found {len(mov_files)} .mov files to convert")
    
    # Convert each .mov file
    for i, mov_file in enumerate(mov_files, 1):
        input_path = os.path.join(input_dir, mov_file)
        
        # Create output filename with .mp4 extension
        output_filename = os.path.splitext(mov_file)[0] + '.mp4'
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"[{i}/{len(mov_files)}] Converting {mov_file} to {output_filename}...")
        
        try:
            # Run the ffmpeg conversion with specific encoding parameters
            (
                ffmpeg
                .input(input_path)
                .output(
                    output_path,
                    # Video settings
                    vcodec='libx264',     # H.264 codec
                    r=30,                 # 30fps frame rate
                    crf=23,               # Constant Rate Factor (quality) - lower is better, 23 is default
                    preset='veryfast',    # Encoding speed preset
                    # Remove audio
                    an=None               # This flag removes audio
                )
                .run(quiet=True, overwrite_output=True)
            )
            print(f"  ✓ Successfully converted to {output_path}")
        except ffmpeg.Error as e:
            print(f"  ✗ Error converting {mov_file}: {e.stderr.decode()}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools.py <input_directory> <output_directory>")
        # poetry run python tools.py test-mov test-mp4
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist or is not a directory")
        sys.exit(1)
    
    convert_all_mov_files(input_dir, output_dir)
    print("Conversion process completed")


