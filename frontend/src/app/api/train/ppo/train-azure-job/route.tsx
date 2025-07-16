import { NextResponse } from 'next/server';

const BASE_URL = process.env.BASE_URL

export async function POST(req: Request) {
  try {
    if (!BASE_URL) {
      throw new Error('BASE_URL environment variable is not set');
    }

    const body = await req.json();

    const response = await fetch(`${BASE_URL}/train/ppo/train-azure-job`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json(
        { error: errorText || 'Unknown error from backend' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (err: any) {
    console.log(err)
    return NextResponse.json(
      { error: err.message || 'Internal Server Error' },
      { status: 500 }
    );
  }
}
