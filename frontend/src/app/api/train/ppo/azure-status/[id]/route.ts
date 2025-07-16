import { NextResponse } from "next/server";

const BASE_URL = process.env.BASE_URL


export async function GET(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const res = await fetch(`${BASE_URL}/train/ppo/azure-status/${id}`);

    if (!res.ok) {
      return NextResponse.json(
        { error: "Failed to fetch status from backend" },
        { status: res.status }
      );
    }

    const data = await res.json();

    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: "Unexpected server error", details: (error as Error).message },
      { status: 500 }
    );
  }
}
