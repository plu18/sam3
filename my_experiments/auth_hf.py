import os
import sys

import huggingface_hub


def authenticate():
    print("Hugging Face Authentication Helper")
    print("----------------------------------")

    # Check if already logged in
    try:
        user = huggingface_hub.whoami()
        print(f"✅ You are already logged in as: {user['name']}")
        print(
            "If you want to re-login, please run this script again and enter a new token."
        )
    except Exception:
        print("❌ You are currently NOT logged in.")

    token = None
    if len(sys.argv) > 1:
        token = sys.argv[1]
        print(f"\nUsing token provided via command line argument.")
    else:
        print("\nTo authenticate, you need a User Access Token from Hugging Face.")
        print("1. Go to https://huggingface.co/settings/tokens")
        print(
            "2. Create a new token (READ access is sufficient for downloading models)."
        )
        print("3. Copy the token and paste it below.")
        print(
            "\n(If you have set the HF_TOKEN environment variable, it will be used automatically)"
        )

        token = input("\nEnter your Hugging Face Token (hidden input): ").strip()

    if not token:
        print("No token entered. Exiting.")
        return

    try:
        huggingface_hub.login(token=token)
        print("\n✅ Successfully logged in!")

        # Verify access to SAM3
        print("\nVerifying access to facebook/sam3...")
        try:
            huggingface_hub.repo_info("facebook/sam3")
            print("✅ You have access to the facebook/sam3 repository.")
        except Exception as e:
            print("⚠️  You are logged in, but cannot access facebook/sam3.")
            print(
                "   Please make sure you have accepted the license agreement at: https://huggingface.co/facebook/sam3"
            )
            print(f"   Error details: {e}")

    except Exception as e:
        print(f"\n❌ Login failed: {e}")


if __name__ == "__main__":
    authenticate()
