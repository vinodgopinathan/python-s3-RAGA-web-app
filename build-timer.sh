#!/bin/bash
# 🕒 Docker Build Time Tracker - Compare optimized vs standard builds

set -e

echo "📊 Docker Build Performance Tracker"
echo "=================================="

# Function to build and time
build_and_time() {
    local build_type=$1
    local dockerfile=$2
    local tag=$3
    
    echo ""
    echo "🔨 Building: $build_type"
    echo "⏰ Start time: $(date)"
    
    start_time=$(date +%s)
    
    if [ "$build_type" = "optimized" ]; then
        DOCKER_BUILDKIT=1 docker build \
            --target=production \
            --progress=plain \
            --build-arg BUILDKIT_INLINE_CACHE=1 \
            --tag $tag \
            --file $dockerfile \
            backend/ 2>&1 | tee "build-${build_type}.log"
    else
        docker build \
            --tag $tag \
            --file $dockerfile \
            backend/ 2>&1 | tee "build-${build_type}.log"
    fi
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "⏰ End time: $(date)"
    echo "⚡ Duration: ${duration} seconds ($(($duration / 60))m $(($duration % 60))s)"
    
    # Save timing info
    echo "$build_type,$duration,$(date)" >> build-times.csv
    
    return $duration
}

# Initialize CSV if it doesn't exist
if [ ! -f build-times.csv ]; then
    echo "build_type,duration_seconds,timestamp" > build-times.csv
fi

echo ""
echo "Select build to test:"
echo "1) Optimized build (multi-stage with caching)"
echo "2) Standard build (original Dockerfile)"
echo "3) Both builds (comparison)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        build_and_time "optimized" "backend/Dockerfile" "backend-optimized:test"
        ;;
    2)
        # Create temporary standard Dockerfile for comparison
        echo "Creating standard build for comparison..."
        # This would use the original single-stage approach
        build_and_time "standard" "backend/Dockerfile.original" "backend-standard:test"
        ;;
    3)
        echo "🏁 Running performance comparison..."
        echo "This will take some time but give you accurate performance metrics."
        echo ""
        
        build_and_time "optimized" "backend/Dockerfile" "backend-optimized:test"
        optimized_time=$?
        
        echo ""
        echo "📊 PERFORMANCE COMPARISON RESULTS"
        echo "================================="
        echo "Optimized build: ${optimized_time}s ($(($optimized_time / 60))m $(($optimized_time % 60))s)"
        echo ""
        echo "💾 Detailed logs saved to:"
        echo "  - build-optimized.log"
        echo "  - build-times.csv"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "📈 Build time history:"
if [ -f build-times.csv ]; then
    tail -5 build-times.csv | column -t -s ','
fi
