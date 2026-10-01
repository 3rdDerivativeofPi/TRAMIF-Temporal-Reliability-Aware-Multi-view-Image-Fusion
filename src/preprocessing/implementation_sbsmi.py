from pathlib import Path

from zipfile import ZipFile



import numpy as np

from PIL import Image





# ---------------------------------------------------------------------------

# SMALL VALIDATION HELPERS

# ---------------------------------------------------------------------------





def _check_bit_num(bit_num: int) -> None:

    """Check the short-bit state length."""

    # The main SBSMI reproduction uses bit_num = 6.

    # We allow 1..8 so the same code can support short-bit ablations later.

    if (

        isinstance(bit_num, bool)

        or not isinstance(bit_num, int)

        or not 1 <= bit_num <= 8

    ):

        raise ValueError("bit_num must be an integer from 1 to 8.")







def _check_chunk_size(chunk_size: int) -> None:

    """Check that each stream read requests a positive number of bytes."""

    if (

        isinstance(chunk_size, bool)

        or not isinstance(chunk_size, int)

        or chunk_size <= 0

    ):

        raise ValueError("chunk_size must be a positive integer.")







def _relative_path(value: str | Path) -> Path:

    """Return a safe relative path for folder/ZIP bulk generation."""

    path = Path(value)



    if path.is_absolute() or ".." in path.parts:

        raise ValueError(

            "Sample paths must be relative and must not contain '..'."

        )



    return path





# ---------------------------------------------------------------------------

# 1. BIT STREAM -> SHORT BIT STATES

# ---------------------------------------------------------------------------





def iterate_lbit_states(

    b_input_strm,

    bit_num: int = 6,

    chunk_size: int = 65536,

):

    """

    Yield non-overlapping bit_num-bit states from a binary stream.



    Example with bit_num = 6:



        11010110 01101100

        -> 110101 | 100110 | 1100

        -> 53, 38, 12



    The stream stays continuous across byte and chunk boundaries.

    """

    _check_bit_num(bit_num)

    _check_chunk_size(chunk_size)



    # Stores bits already read but not yet used in a complete state.

    # We use an integer instead of a list because Python bit operations

    # can join/extract bits directly and efficiently.

    bit_buffer = 0



    # Number of valid unused bits currently represented by bit_buffer.

    # This matters because integers do not preserve leading zeroes.

    bit_count = 0



    # Example for bit_num = 6:

    #   (1 << 6) - 1 = 63 = binary 111111

    # This lets us keep exactly 6 bits when extracting one state.

    state_mask = (1 << bit_num) - 1



    while True:

        # Read up to chunk_size BYTES from the current stream position.

        # At end-of-file, read() returns b"".

        chunk = b_input_strm.read(chunk_size)



        if not chunk:

            break



        # Iterating over bytes gives integers from 0 to 255.

        for byte in chunk:

            # Put the new byte after the unused bits already in bit_buffer.

            #

            # Example:

            #   old unused bits: 10

            #   new byte:        01101100

            #   result:          1001101100

            #

            # << 8 makes room for one byte; | byte fills that space.

            bit_buffer = (bit_buffer << 8) | byte

            bit_count += 8



            # One new byte may give us enough bits for multiple states.

            while bit_count >= bit_num:

                # Number of bits that will remain after taking one state.

                remaining_bits = bit_count - bit_num



                # Move the next state to the right and keep bit_num bits.

                # Example:

                #   buffer = 1001101100

                #   remaining_bits = 4

                #   state = 100110 = 38

                state = (

                    bit_buffer >> remaining_bits

                ) & state_mask



                # Yield one state without ending the generator.

                yield state



                # Remove the state we just consumed from bit_buffer.

                if remaining_bits == 0:

                    bit_buffer = 0

                else:

                    # Keep only the lowest remaining_bits bits.

                    remaining_mask = (1 << remaining_bits) - 1

                    bit_buffer &= remaining_mask



                # Keep the count synchronized with the buffer contents.

                bit_count = remaining_bits



    # End of the WHOLE STREAM. There may be one short final state.

    if bit_count > 0:

        # IMPORTANT: do NOT left-shift the tail.

        #

        # Source-compatible rule:

        # missing HIGH-order bits are treated as zero.

        #

        # Example with bit_num = 6:

        #   leftover bits = 101

        #   interpretation = 000101 = 5

        #

        # bit_buffer already stores integer 5, so we yield it directly.

        # Left-shifting would create 101000 = 40, which adds zeroes on

        # the wrong side and changes the representation.

        yield bit_buffer





# ---------------------------------------------------------------------------

# 2. SHORT BIT STATES -> SBSMI MATRIX

# ---------------------------------------------------------------------------





def stream_to_sbsmi(

    stream,

    bit_num: int = 6,

    chunk_size: int = 65536,

) -> np.ndarray:

    """Convert one readable binary stream into an SBSMI uint8 matrix."""

    _check_bit_num(bit_num)

    _check_chunk_size(chunk_size)



    # bit_num = 6 -> 2^6 = 64 possible states -> 64 x 64 SBSMI.

    n_states = 1 << bit_num



    # counts[i, j] = how many times state i is followed by state j.

    # uint64 avoids overflow on real binaries with many transitions.

    counts = np.zeros(

        (n_states, n_states),

        dtype=np.uint64,

    )



    previous_state = None

    number_of_states = 0



    # Consume states one at a time so we do not store the whole file's

    # state sequence in memory.

    for current_state in iterate_lbit_states(

        stream,

        bit_num=bit_num,

        chunk_size=chunk_size,

    ):

        number_of_states += 1



        # The first state has no previous state, so no transition yet.

        if previous_state is not None:

            counts[

                previous_state,

                current_state,

            ] += 1



        previous_state = current_state



    if number_of_states == 0:

        raise ValueError(

            "Cannot construct SBSMI from empty input."

        )



    # Row-normalized transition probabilities are stored here.

    probabilities = np.zeros(

        (n_states, n_states),

        dtype=np.float64,

    )



    # Normalize EACH ROW separately because SBSMI uses:

    #   P(next_state = j | current_state = i)

    for row in range(n_states):

        row_total = counts[row].sum()



        if row_total > 0:

            probabilities[row] = (

                counts[row] / row_total

            )

        # If row_total == 0, this state has no outgoing transitions,

        # so the row correctly remains all zeros.



    # Source-compatible quantization:

    #   pixel = floor(P * 255)

    # Use floor(), not round().

    image = np.floor(

        probabilities * 255

    ).astype(np.uint8)



    return image





# ---------------------------------------------------------------------------

# 3. NORMAL FILE WRAPPER

# ---------------------------------------------------------------------------





def file_to_sbsmi(

    path: str | Path,

    bit_num: int = 6,

    chunk_size: int = 65536,

) -> np.ndarray:

    """Generate one SBSMI from a normal file on disk."""

    # This wrapper only opens the file. The actual SBSMI algorithm stays

    # in stream_to_sbsmi() so there is only one source of truth.

    with Path(path).open("rb") as stream:

        return stream_to_sbsmi(

            stream,

            bit_num=bit_num,

            chunk_size=chunk_size,

        )





# ---------------------------------------------------------------------------

# 4. ZIP ENTRY WRAPPER

# ---------------------------------------------------------------------------





def zip_entry_to_sbsmi(

    zip_file: ZipFile,

    member_path: str,

    bit_num: int = 6,

    chunk_size: int = 65536,

) -> np.ndarray:

    """Generate one SBSMI from one member inside an already-open ZIP."""

    # The whole ZIP is NOT extracted. Only this member is opened as a

    # readable stream and sent into the same stream_to_sbsmi() core.

    with zip_file.open(

        member_path,

        "r",

    ) as stream:

        return stream_to_sbsmi(

            stream,

            bit_num=bit_num,

            chunk_size=chunk_size,

        )





# ---------------------------------------------------------------------------

# 5. VALIDATE ONE SBSMI

# ---------------------------------------------------------------------------





def validate_sbsmi(

    image: np.ndarray,

    bit_num: int = 6,

) -> None:

    """Check the expected SBSMI shape, dtype, and value range."""

    _check_bit_num(bit_num)



    if not isinstance(image, np.ndarray):

        raise TypeError(

            "SBSMI must be a NumPy array."

        )



    expected_size = 1 << bit_num

    expected_shape = (

        expected_size,

        expected_size,

    )



    if image.shape != expected_shape:

        raise ValueError(

            "Invalid SBSMI shape: "

            f"expected {expected_shape}, "

            f"got {image.shape}."

        )



    if image.dtype != np.uint8:

        raise ValueError(

            "Invalid SBSMI dtype: "

            f"expected uint8, "

            f"got {image.dtype}."

        )



    # uint8 already guarantees 0..255, but this explicit check documents

    # the intended representation contract.

    if image.min() < 0 or image.max() > 255:

        raise ValueError(

            "SBSMI pixel values must be in [0, 255]."

        )





# ---------------------------------------------------------------------------

# 6. SAVE ONE SBSMI

# ---------------------------------------------------------------------------





def save_sbsmi(

    image: np.ndarray,

    output_path: str | Path,

    bit_num: int = 6,

) -> None:

    """Validate and save one SBSMI as a grayscale PNG."""

    validate_sbsmi(

        image,

        bit_num=bit_num,

    )



    output_path = Path(output_path)



    # Create parent folders automatically if needed.

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )



    # A 2-D uint8 NumPy array is interpreted by Pillow as grayscale.

    Image.fromarray(image).save(

        output_path

    )





# ---------------------------------------------------------------------------

# HELPERS FOR BULK DATASET GENERATION

# ---------------------------------------------------------------------------





def _output_path_for_sample(

    output_root: Path,

    sample: dict,

) -> Path:

    """Choose the PNG output path for one sample."""

    relative_input = _relative_path(

        sample["path"]

    )



    sha = str(

        sample.get("sha") or ""

    ).strip()



    family = str(

        sample.get("family") or ""

    ).strip()



    # Prefer SHA as filename when metadata provides one.

    if sha:

        filename = f"{sha}.png"

    else:

        filename = f"{relative_input.stem}.png"



    # If family is known, place the image under a family folder.

    if family:

        return (

            output_root

            / family

            / filename

        )



    # SHA but no family: save directly under output_root.

    if sha:

        return output_root / filename



    # No SHA/family metadata: preserve relative input structure.

    return (

        output_root

        / relative_input.with_suffix(".png")

    )





# ---------------------------------------------------------------------------

# 7. BULK DATASET GENERATION

# ---------------------------------------------------------------------------





def generate_sbsmi_dataset(

    source_path: str | Path,

    output_root: str | Path,

    source_mode: str,

    samples: list[dict] | None = None,

    bit_num: int = 6,

    chunk_size: int = 65536,

    overwrite: bool = False,

    max_files: int | None = None,

) -> list[dict]:

    """

    Generate many SBSMIs from either a folder or a ZIP archive.



    source_mode="folder":

        - samples=None -> recursively process every file in source_path

        - samples given -> process only listed relative paths



    source_mode="zip":

        - samples is required

        - sample["path"] is the member path inside the ZIP

        - optional sample["sha"] and sample["family"] control naming



    Returns one status dictionary per requested sample.

    """

    _check_bit_num(bit_num)

    _check_chunk_size(chunk_size)



    source_path = Path(source_path)

    output_root = Path(output_root)



    # Accept harmless variations such as "ZIP" or " folder ".

    source_mode = source_mode.lower().strip()



    if source_mode not in {

        "folder",

        "zip",

    }:

        raise ValueError(

            "source_mode must be 'folder' or 'zip'."

        )



    # ------------------------------------------------------------------

    # Prepare the list of samples.

    # ------------------------------------------------------------------



    # Optional safety limit.

    # Example: max_files=200 means the function refuses to process more

    # than 200 samples. None means no explicit limit.

    if (

        max_files is not None

        and (

            isinstance(max_files, bool)

            or not isinstance(max_files, int)

            or max_files <= 0

        )

    ):

        raise ValueError(

            "max_files must be a positive integer or None."

        )



    if source_mode == "folder":

        if not source_path.is_dir():

            raise ValueError(

                "source_path must be a folder in folder mode."

            )



        # No explicit sample list: discover every file recursively first.

        # IMPORTANT: discovery only counts paths; SBSMI generation has not

        # started yet, so max_files can stop the job safely.

        if samples is None:

            discovered_files = [

                path

                for path in sorted(

                    source_path.rglob("*")

                )

                if path.is_file()

            ]



            file_count = len(discovered_files)



            print()

            print(

                f"Folder mode: found {file_count} file(s) under:"

            )

            print(source_path)



            if file_count == 0:

                raise RuntimeError(

                    "No files were found in the input folder."

                )



            if (

                max_files is not None

                and file_count > max_files

            ):

                raise RuntimeError(

                    f"Found {file_count} files, "

                    f"which exceeds max_files={max_files}. "

                    "Check source_path before continuing."

                )



            samples = [

                {

                    "path": str(

                        path.relative_to(source_path)

                    ),

                }

                for path in discovered_files

            ]



        # Explicit sample list: count and validate it too.

        else:

            sample_count = len(samples)



            print()

            print(

                f"Folder mode: {sample_count} selected sample(s)."

            )



            if sample_count == 0:

                raise RuntimeError(

                    "The sample list is empty."

                )



            if (

                max_files is not None

                and sample_count > max_files

            ):

                raise RuntimeError(

                    f"Received {sample_count} samples, "

                    f"which exceeds max_files={max_files}."

                )



    else:

        # ZIP mode.

        if not source_path.is_file():

            raise ValueError(

                "source_path must be a ZIP file in zip mode."

            )



        # ZIP mode deliberately requires an explicit cohort/sample list.

        # This prevents accidentally processing an entire huge archive and

        # keeps experimental sample selection explicit.

        if samples is None:

            raise ValueError(

                "samples is required in zip mode so the intended cohort "

                "is selected explicitly."

            )



        sample_count = len(samples)



        print()

        print(

            f"ZIP mode: {sample_count} selected sample(s)."

        )



        if sample_count == 0:

            raise RuntimeError(

                "The sample list is empty."

            )



        if (

            max_files is not None

            and sample_count > max_files

        ):

            raise RuntimeError(

                f"Received {sample_count} samples, "

                f"which exceeds max_files={max_files}."

            )



    # Every requested sample receives one result record.

    records: list[dict] = []



    # Folder mode does not need a ZipFile object.

    # ZIP mode opens the archive ONCE for the whole bulk job.

    zip_file: ZipFile | None = (

        ZipFile(

            source_path,

            "r",

        )

        if source_mode == "zip"

        else None

    )



    try:

        for sample in samples:

            # Path is relative to either the source folder or ZIP root.

            relative_input = _relative_path(

                sample["path"]

            )



            output_path = _output_path_for_sample(

                output_root,

                sample,

            )



            # Shared information for success/skipped/failed records.

            base_record = {

                "path": str(relative_input),

                "sha": str(sample.get("sha") or ""),

                "family": str(sample.get("family") or ""),

                "output_path": str(output_path),

            }



            # Resume support: do not regenerate existing images unless

            # overwrite=True.

            if output_path.exists() and not overwrite:

                records.append({

                    **base_record,

                    "status": "skipped",

                    "error_type": "",

                    "error_message": "",

                })

                continue



            try:

                if source_mode == "folder":

                    # Normal-file mode.

                    image = file_to_sbsmi(

                        source_path / relative_input,

                        bit_num=bit_num,

                        chunk_size=chunk_size,

                    )



                else:

                    # Pylance sees zip_file as ZipFile | None because

                    # folder mode uses None. This branch only runs in ZIP

                    # mode, so here it must be a real ZipFile.

                    assert zip_file is not None



                    image = zip_entry_to_sbsmi(

                        zip_file,

                        relative_input.as_posix(),

                        bit_num=bit_num,

                        chunk_size=chunk_size,

                    )



                # Validate before writing anything to disk.

                validate_sbsmi(

                    image,

                    bit_num=bit_num,

                )



                save_sbsmi(

                    image,

                    output_path,

                    bit_num=bit_num,

                )



                records.append({

                    **base_record,

                    "status": "success",

                    "error_type": "",

                    "error_message": "",

                })



            except Exception as exc:

                # One bad file should not kill the whole bulk job.

                # BUT the failure must be recorded instead of silently

                # skipped so preprocessing remains auditable.

                records.append({

                    **base_record,

                    "status": "failed",

                    "error_type": type(exc).__name__,

                    "error_message": str(exc),

                })



    finally:

        # Close the ZIP once after all requested members are done.

        if zip_file is not None:

            zip_file.close()



    return records
