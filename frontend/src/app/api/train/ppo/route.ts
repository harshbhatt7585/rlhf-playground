import { NextResponse } from 'next/server'

const BASE_URL = process.env.BASE_URL

export async function POST(req: Request) {
  try {
    const body = await req.json()

    const response = await fetch(`${BASE_URL}/train/ppo`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const error = await response.text()
      return NextResponse.json({ error }, { status: response.status })
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (err: any) {
    return NextResponse.json({ error: err.message || 'Internal Server Error' }, { status: 500 })
  }
}
