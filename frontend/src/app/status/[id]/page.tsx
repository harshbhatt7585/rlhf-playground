'use client'

import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";

interface Status {
  job_id: string;
  status: string;
//   completed: boolean;
}

function Status() {
  const [status, setStatus] = useState<Status | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const params = useParams();
  const id = (params as { id: string }).id;

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`/api/train/ppo/azure-status/${id}`);
        if (!res.ok) {
          throw new Error("Something went wrong while fetching status.");
        }
        const data = await res.json();
        setStatus(data);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (id) {
      fetchStatus();
    }
  }, [id]);

  return (
    <div className="p-4">
      {loading && <p>Loading...</p>}
      {error && <p className="text-red-500">Error: {error}</p>}
      {status && (
        <div>
          <h2 className="text-xl font-semibold">Job Status</h2>
          <p><strong>Job ID:</strong> {status.job_id}</p>
          <p><strong>Status:</strong> {status.status}</p>
          <p><strong>Completed:</strong> {status.completed ? "Yes" : "No"}</p>
        </div>
      )}
    </div>
  );
}

export default Status;
