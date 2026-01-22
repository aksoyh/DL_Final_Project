# Test script - extracted from notebook for debugging
# Run with: julia --project=. test_notebook.jl

println("="^60)
println("TESTING NOTEBOOK CODE")
println("="^60)

# ============================================
# CELL 1: Install and import packages
# ============================================
println("\n[1/5] Installing and loading packages...")

using Pkg

# List of required packages
required_packages = [
    "Flux",
    "Images",
    "Plots",
    "ProgressMeter",
    "BSON",
    "ImageMagick"
]

# Check and install missing packages
installed_packages = [pkg.name for pkg in values(Pkg.dependencies())]

for pkg in required_packages
    if !(pkg in installed_packages)
        println("Installing $pkg...")
        Pkg.add(pkg)
    else
        println("$pkg is already installed.")
    end
end

println("\nAll packages ready! Loading...")

# Import required packages
using Flux
using Flux: onehotbatch, onecold, crossentropy
using Images
using Plots
using Statistics
using Random
using ProgressMeter
using BSON: @save, @load

# Set random seed for reproducibility
Random.seed!(42)

println("Packages loaded successfully!")

# ============================================
# CELL 2: Define paths and constants
# ============================================
println("\n[2/5] Defining configuration...")

const DATA_DIR = "data"
const TRAIN_DIR = joinpath(DATA_DIR, "train")
const TEST_DIR = joinpath(DATA_DIR, "test")

const CLASSES = ["Cars", "Memes", "Mountains", "Selfies", "Trees", "Whatsapp_Screenshots"]
const NUM_CLASSES = length(CLASSES)

# Image parameters
const IMG_SIZE = (128, 128)
const CHANNELS = 3

println("Configuration:")
println("  Classes: ", CLASSES)
println("  Number of classes: ", NUM_CLASSES)
println("  Image size: ", IMG_SIZE)

# ============================================
# CELL 3: Define CNN Model
# ============================================
println("\n[3/5] Building CNN model...")

function build_cnn_model()
    return Chain(
        # First Convolutional Block
        Conv((3, 3), 3 => 32, relu, pad=SamePad()),
        MaxPool((2, 2)),

        # Second Convolutional Block
        Conv((3, 3), 32 => 64, relu, pad=SamePad()),
        MaxPool((2, 2)),

        # Third Convolutional Block
        Conv((3, 3), 64 => 128, relu, pad=SamePad()),
        MaxPool((2, 2)),

        # Flatten and Dense Layers
        Flux.flatten,
        Dense(128 * 16 * 16, 256, relu),
        Dropout(0.5),
        Dense(256, NUM_CLASSES),
        softmax
    )
end

# Create model instance
model = build_cnn_model()
println("Model created:")
println(model)

# ============================================
# CELL 4: Test model with dummy input
# ============================================
println("\n[4/5] Testing model with dummy input...")

dummy_input = rand(Float32, IMG_SIZE[1], IMG_SIZE[2], CHANNELS, 1)
dummy_output = model(dummy_input)
println("Input shape: ", size(dummy_input))
println("Output shape: ", size(dummy_output))
println("Output probabilities sum to: ", sum(dummy_output))

# ============================================
# CELL 5: Check data directory
# ============================================
println("\n[5/5] Checking data directories...")

function count_images(dir)
    if !isdir(dir)
        return 0
    end
    files = filter(f -> lowercase(splitext(f)[2]) in [".jpg", ".jpeg", ".png"], readdir(dir))
    return length(files)
end

total_images = 0
for class_name in CLASSES
    train_path = joinpath(TRAIN_DIR, class_name)
    test_path = joinpath(TEST_DIR, class_name)
    train_count = count_images(train_path)
    test_count = count_images(test_path)
    global total_images += train_count + test_count
    println("  $class_name: train=$train_count, test=$test_count")
end

println("\nTotal images found: $total_images")

if total_images == 0
    println("\n⚠️  WARNING: No images found in data directory!")
    println("Please download the dataset from Kaggle:")
    println("https://www.kaggle.com/datasets/n0obcoder/mobile-gallery-image-classification-data")
else
    println("\n✅ Dataset is ready!")
end

println("\n" * "="^60)
println("TEST COMPLETE - All code cells executed successfully!")
println("="^60)
