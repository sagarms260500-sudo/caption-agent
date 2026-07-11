name: Build and Push Caption Agent

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build and push (linux/amd64)
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          platforms: linux/amd64
          tags: ${{ secrets.DOCKERHUB_USERNAME }}/caption-agent:latest
          build-args: |
            GEMINI_API_KEY=${{ secrets.GEMINI_API_KEY }}
            OPENROUTER_API_KEY=${{ secrets.OPENROUTER_API_KEY }}
            ANTHROPIC_API_KEY=${{ secrets.ANTHROPIC_API_KEY }}
            MOONSHOT_API_KEY=${{ secrets.MOONSHOT_API_KEY }}
